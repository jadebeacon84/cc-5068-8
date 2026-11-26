"""Editable dubbing cue sheet with reversible cut/restore editing.

The cue sheet is the editable surface around kernel events. Operators can cut
a small section between two bell rings and later restore it; the sheet reacts
by merging or re-splitting the affected markers. Every operation is logged so
the dry source tracks can later confirm the original event identity and exact
boundaries.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from .events import SoundEvent, dicts_to_events, events_to_dicts, merge_events, split_event


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class CueSheet:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, list[dict]] = {}
        self.history: list[dict] = []
        self.cuts: dict[str, list[dict]] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.entries = raw.get("entries", {})
            self.history = raw.get("history", [])
            self.cuts = raw.get("cuts", {})

    # ------------------------------------------------------------------ I/O
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "version": 1,
            "updated_at": _now(),
            "entries": self.entries,
            "cuts": self.cuts,
            "history": self.history,
        }, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------- seeding
    def replace_from_events(self, recording_id: str, events: list[SoundEvent],
                            source: str = "kernel") -> None:
        rows = events_to_dicts(events)
        for row in rows:
            row["cue_status"] = "predicted"
            row["edit_source"] = source
        self.entries[recording_id] = rows
        self._log("seed", recording_id, {"count": len(rows), "source": source})
        self.save()

    def get_events(self, recording_id: str) -> list[SoundEvent]:
        return dicts_to_events(self.entries.get(recording_id, []))

    def set_events(self, recording_id: str, events: list[SoundEvent],
                   status: str = "edited", source: str = "operator") -> list[dict]:
        rows = events_to_dicts(events)
        for row in rows:
            row.setdefault("cue_status", status)
            row["edit_source"] = source
        self.entries[recording_id] = rows
        self.save()
        return rows

    # ------------------------------------------------------------- editing
    def relabel(self, recording_id: str, uid: str, new_class: str) -> dict:
        rows = self.entries[recording_id]
        for row in rows:
            if row["event_uid"] == uid:
                old = row["cls"]
                row["cls"] = new_class
                row["cue_status"] = "edited"
                row["edit_source"] = "operator"
                self._log("relabel", recording_id,
                          {"uid": uid, "from": old, "to": new_class})
                self.save()
                return row
        raise KeyError(uid)

    def delete(self, recording_id: str, uid: str) -> None:
        before = len(self.entries[recording_id])
        self.entries[recording_id] = [
            row for row in self.entries[recording_id] if row["event_uid"] != uid
        ]
        if len(self.entries[recording_id]) == before:
            raise KeyError(uid)
        self._log("delete", recording_id, {"uid": uid})
        self.save()

    def merge(self, recording_id: str, uid_a: str, uid_b: str) -> dict:
        events = self.get_events(recording_id)
        merged = merge_events(events, uid_a, uid_b)
        if merged is None:
            raise ValueError("merge requires two existing same-class events")
        remaining = [e for e in events
                     if e.event_uid not in (uid_a, uid_b)]
        remaining.append(merged)
        remaining.sort(key=lambda e: (e.start, e.end))
        self._log("merge", recording_id, {"a": uid_a, "b": uid_b})
        return self.set_events(recording_id, remaining, source="operator")[0]

    def split(self, recording_id: str, uid: str, at: float) -> list[dict]:
        events = self.get_events(recording_id)
        result = split_event(events, uid, at)
        if result is None:
            raise ValueError("split point must lie inside the event")
        left, right = result
        remaining = [e for e in events if e.event_uid != uid]
        remaining.extend([left, right])
        remaining.sort(key=lambda e: (e.start, e.end))
        self._log("split", recording_id, {"uid": uid, "at": at})
        return self.set_events(recording_id, remaining, source="operator")

    # ------------------------------------------- reversible cut and restore
    def cut_section(self, recording_id: str, start: float, end: float) -> dict:
        """Remove [start, end] from the cue timeline and merge markers whose
        gap is fully contained in the removed section."""
        if end <= start:
            raise ValueError("cut end must be greater than start")
        rows = deepcopy(self.entries.get(recording_id, []))
        snapshot = deepcopy(rows)
        shift = end - start
        affected = []
        for row in rows:
            a, b = row["start"], row["end"]
            if b <= start:
                # entirely before the cut: unchanged
                continue
            if a >= end:
                # entirely after the cut: shift left
                row["start"] = round(a - shift, 3)
                row["end"] = round(b - shift, 3)
            elif a >= start and b <= end:
                # fully swallowed by the cut: collapse to the cut point
                row["start"] = round(start, 3)
                row["end"] = round(start, 3)
            elif a < start and b <= end:
                # starts before, ends inside the cut: trim at cut start
                row["end"] = round(start, 3)
            elif a >= start and b > end:
                # starts inside, ends after: keep tail, anchored at cut start
                row["start"] = round(start, 3)
                row["end"] = round(start + max(0.0, b - end), 3)
                assert abs((row["end"] - row["start"]) -
                           max(0.0, b - end)) < 1e-6
            else:
                # spans the whole cut: shrink by the removed duration
                row["end"] = round(b - shift, 3)
            affected.append(row["event_uid"])

        # Merge same-class neighbors now touching across the cut.
        merged_rows = _merge_touching(rows, gap_tol=0.05)
        self.cuts.setdefault(recording_id, []).append({
            "start": start, "end": end,
            "cut_at": _now(),
            "affected_uids": affected,
            "snapshot": snapshot,
            "restored": False,
        })
        self.entries[recording_id] = merged_rows
        self._log("cut", recording_id, {"start": start, "end": end})
        self.save()
        return {"entries": merged_rows, "cut": self.cuts[recording_id][-1]}

    def restore_section(self, recording_id: str, cut_index: int = -1) -> dict:
        cuts = self.cuts.get(recording_id, [])
        if not cuts:
            raise KeyError("no cut to restore")
        cut = cuts[cut_index]
        if cut.get("restored"):
            raise ValueError("cut already restored")
        self.entries[recording_id] = deepcopy(cut["snapshot"])
        cut["restored"] = True
        cut["restored_at"] = _now()
        self._log("restore", recording_id,
                  {"start": cut["start"], "end": cut["end"]})
        self.save()
        return {"entries": self.entries[recording_id], "cut": cut}

    # ------------------------------------------------------------ feedback
    def mark_failure(self, recording_id: str, uid: str,
                     reason: str, expected_class: str | None = None) -> dict:
        for row in self.entries.get(recording_id, []):
            if row["event_uid"] == uid:
                row["cue_status"] = "failure"
                row["failure_reason"] = reason
                if expected_class:
                    row["expected_class"] = expected_class
                self._log("failure", recording_id,
                          {"uid": uid, "reason": reason})
                self.save()
                return row
        raise KeyError(uid)

    def verify_with_sources(self, recording_id: str,
                            source_events: list[dict],
                            iou_threshold: float = 0.3) -> dict:
        """Check edited cue rows against dry source-track ground truth."""
        rows = self.entries.get(recording_id, [])
        report = {"matched": [], "unmatched_cues": [], "missing_sources": []}
        used = set()
        for row in rows:
            best, best_iou = None, 0.0
            for g in source_events:
                if g["event_id"] in used:
                    continue
                iou = _span_iou(row["start"], row["end"], g["start"], g["end"])
                if iou > best_iou:
                    best, best_iou = g, iou
            if best is not None and best_iou >= iou_threshold:
                used.add(best["event_id"])
                report["matched"].append({
                    "uid": row["event_uid"],
                    "source_event_id": best["event_id"],
                    "class_match": row["cls"] == best["class"],
                    "iou": round(best_iou, 3),
                    "boundary_error": {
                        "start": round(abs(row["start"] - best["start"]), 3),
                        "end": round(abs(row["end"] - best["end"]), 3),
                    },
                })
            else:
                report["unmatched_cues"].append(row["event_uid"])
        gt_ids = {g["event_id"] for g in source_events}
        report["missing_sources"] = sorted(gt_ids - used)
        report["all_identities_exact"] = (
            not report["unmatched_cues"] and not report["missing_sources"]
            and all(m["class_match"] and m["iou"] >= iou_threshold
                    for m in report["matched"])
        )
        self._log("verify", recording_id,
                  {"exact": report["all_identities_exact"]})
        self.save()
        return report

    def _log(self, action: str, recording_id: str, detail: dict) -> None:
        self.history.append({
            "time": _now(), "action": action,
            "recording_id": recording_id, "detail": detail,
        })


def _span_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, max(a1, b1) - min(a0, b0))
    return inter / union


def _merge_touching(rows: list[dict], gap_tol: float) -> list[dict]:
    if not rows:
        return rows
    rows = sorted(
        (deepcopy(r) for r in rows if r["end"] - r["start"] > 1e-6),
        key=lambda r: (r["start"], r["end"]),
    )
    if not rows:
        return []
    out = [deepcopy(rows[0])]
    for row in rows[1:]:
        prev = out[-1]
        if row["cls"] == prev["cls"] and row["start"] - prev["end"] <= gap_tol:
            prev["end"] = max(prev["end"], row["end"])
            prev["possible_count"] = prev.get("possible_count", 1) + row.get("possible_count", 1)
            prev["cue_status"] = "merged_by_cut"
            prev["note"] = (prev.get("note", "") + "; merged after cut").strip("; ")
        else:
            out.append(deepcopy(row))
    return out
