"""Event-level evaluation. Adjudication happens per event, never per frame.

A predicted event is a true positive only when it matches one ground-truth
event of the same class with IoU >= threshold. Unmatched predictions are false
alarms; unmatched ground truth are misses. Unknown/background intervals have
their own false-alarm accounting and rejection rate so "reject unknown" and
"don't equate background with a target" are measured explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TARGET_CLASSES, UNKNOWN_CLASS
from .events import SoundEvent


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, max(a1, b1) - min(a0, b0))
    return inter / union


def match_events(pred: list[SoundEvent], gt: list[dict],
                 iou_threshold: float = 0.3) -> dict:
    """Bipartite matching restricted to the same target class."""
    result = {"tp": [], "fp": [], "fn": [], "boundary_errors": []}
    gt_used = set()
    for cls in TARGET_CLASSES:
        p = [e for e in pred if e.cls == cls]
        g = [x for x in gt if x["class"] == cls]
        if not p:
            for x in g:
                result["fn"].append({"class": cls, "gt": x})
            continue
        if not g:
            for e in p:
                result["fp"].append({"event": e.to_dict(), "reason": "no_ground_truth"})
            continue
        cost = np.array([[1.0 - _iou(e.start, e.end, x["start"], x["end"])
                          for x in g] for e in p])
        rows, cols = linear_sum_assignment(cost)
        matched_g = set()
        for r, c in zip(rows, cols):
            iou = 1.0 - cost[r, c]
            e, x = p[r], g[c]
            if iou >= iou_threshold:
                matched_g.add(c)
                gt_used.add(x["event_id"])
                result["tp"].append({
                    "class": cls,
                    "pred": e.to_dict(),
                    "gt_event_id": x["event_id"],
                    "iou": round(float(iou), 3),
                    "start_error": round(abs(e.start - x["start"]), 3),
                    "end_error": round(abs(e.end - x["end"]), 3),
                })
                result["boundary_errors"].append(
                    (abs(e.start - x["start"]), abs(e.end - x["end"]))
                )
            else:
                result["fp"].append({"event": e.to_dict(),
                                     "reason": "iou_below_threshold"})
        for c, x in enumerate(g):
            if c not in matched_g:
                result["fn"].append({"class": cls, "gt": x})
        for r, e in enumerate(p):
            if r not in rows.tolist():
                result["fp"].append({"event": e.to_dict(), "reason": "unmatched"})
    return result


def _overlaps_any(ev: SoundEvent, spans: list[dict], min_overlap: float = 0.2) -> bool:
    return any(
        min(ev.end, x["end"]) - max(ev.start, x["start"]) >= min_overlap
        for x in spans
    )


def evaluate_recording(pred: list[SoundEvent], labels: dict,
                       iou_threshold: float = 0.3) -> dict:
    gt_targets = [e for e in labels["events"] if e["class"] in TARGET_CLASSES]
    gt_unknown = [e for e in labels["events"] if e["class"] == UNKNOWN_CLASS]
    matching = match_events(pred, gt_targets, iou_threshold)

    false_alarms = list(matching["fp"])
    for fp in false_alarms:
        ev_dict = fp["event"]
        ev = SoundEvent(**{k: v for k, v in ev_dict.items()
                           if k in SoundEvent.__dataclass_fields__})
        if _overlaps_any(ev, gt_unknown):
            fp["reason"] = "unknown_sound_not_rejected"
        elif not _overlaps_any(ev, labels["events"]):
            fp["reason"] = "background_false_alarm"

    # Unknown rejection: fraction of unknown intervals with no overlapping
    # target prediction at all.
    unknown_total = len(gt_unknown)
    unknown_rejected = 0
    for span in gt_unknown:
        hit = any(
            min(e.end, span["end"]) - max(e.start, span["start"]) >= 0.2
            for e in pred
        )
        if not hit:
            unknown_rejected += 1

    return {
        "recording_id": labels["recording_id"],
        "condition": labels.get("condition"),
        "tp": matching["tp"],
        "fp": false_alarms,
        "fn": matching["fn"],
        "unknown_total": unknown_total,
        "unknown_rejected": unknown_rejected,
        "boundary_errors": matching["boundary_errors"],
    }


def summarize(per_recording: list[dict]) -> dict:
    tp = sum(len(r["tp"]) for r in per_recording)
    fp = sum(len(r["fp"]) for r in per_recording)
    fn = sum(len(r["fn"]) for r in per_recording)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_class: dict[str, dict] = {}
    for cls in TARGET_CLASSES:
        ctp = sum(1 for r in per_recording for t in r["tp"] if t["class"] == cls)
        cfp = sum(1 for r in per_recording for f in r["fp"]
                  if f["event"]["cls"] == cls)
        cfn = sum(1 for r in per_recording for f in r["fn"] if f["class"] == cls)
        p = ctp / (ctp + cfp) if ctp + cfp else 0.0
        rcl = ctp / (ctp + cfn) if ctp + cfn else 0.0
        per_class[cls] = {
            "tp": ctp, "fp": cfp, "fn": cfn,
            "precision": round(p, 4),
            "recall": round(rcl, 4),
            "f1": round(2 * p * rcl / (p + rcl), 4) if p + rcl else 0.0,
        }

    boundary = np.array(
        [be for r in per_recording for be in r["boundary_errors"]],
        dtype=np.float64,
    )
    if boundary.size:
        boundary_summary = {
            "mean_start_error": round(float(boundary[:, 0].mean()), 3),
            "mean_end_error": round(float(boundary[:, 1].mean()), 3),
            "median_total_error": round(
                float(np.median(boundary.sum(axis=1))), 3),
        }
    else:
        boundary_summary = {"mean_start_error": None,
                            "mean_end_error": None,
                            "median_total_error": None}

    unknown_total = sum(r["unknown_total"] for r in per_recording)
    unknown_rejected = sum(r["unknown_rejected"] for r in per_recording)
    background_fa = sum(
        1 for r in per_recording for f in r["fp"]
        if f["reason"] == "background_false_alarm"
    )
    unknown_fa = sum(
        1 for r in per_recording for f in r["fp"]
        if f["reason"] == "unknown_sound_not_rejected"
    )

    by_condition: dict[str, dict] = {}
    for r in per_recording:
        cond = r.get("condition") or "unknown"
        bucket = by_condition.setdefault(cond, {"tp": 0, "fp": 0, "fn": 0})
        bucket["tp"] += len(r["tp"])
        bucket["fp"] += len(r["fp"])
        bucket["fn"] += len(r["fn"])
    for cond, b in by_condition.items():
        p = b["tp"] / (b["tp"] + b["fp"]) if b["tp"] + b["fp"] else 0.0
        rc = b["tp"] / (b["tp"] + b["fn"]) if b["tp"] + b["fn"] else 0.0
        b["f1"] = round(2 * p * rc / (p + rc), 4) if p + rc else 0.0

    return {
        "event_level": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_class": per_class,
        "boundary": boundary_summary,
        "false_alarms": {
            "total": fp,
            "background": background_fa,
            "unknown_not_rejected": unknown_fa,
        },
        "unknown_rejection": {
            "total": unknown_total,
            "rejected": unknown_rejected,
            "rejection_rate": round(unknown_rejected / unknown_total, 4)
            if unknown_total else None,
        },
        "by_condition": by_condition,
    }


def write_report(path: Path, per_recording: list[dict],
                 summary: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "adjudication": "event-level bipartite IoU matching; no frame scoring",
        "summary": summary,
        "per_recording": per_recording,
        "extra": extra or {},
    }, indent=2, ensure_ascii=False))
