"""Reusable label-material checks shared by the CLI and the test suite.

The checker reads only on-disk artifacts (manifest.json, labels.json, WAV
files) and never mutates data, so it can run against a clean directory without
hand-edited state. It covers:

  * label duration / boundary sanity, including short-event accounting;
  * recording-level split relations (no recording in both train and eval);
  * file digests (missing/changed WAV files, duplicate source material);
  * legacy one-to-many mapping coverage and unresolved split intervals;
  * the training-pack rule — pending/unclaimed intervals must not contribute
    positive single-class windows, and unclaimed intervals must correspond to
    source tracks that are actually off disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    AUDIT_AUDITED,
    AUDIT_UNCLAIMED,
    TARGET_CLASSES,
    UNKNOWN_CLASS,
)
from .dataset import assert_no_leakage, label_windows, read_manifest
from .features import frame_times, window_starts
from .labeling import (
    check_legacy_coverage,
    find_duplicate_material,
    is_trainable_event,
    recording_digests,
)

# Hard floor: any interval shorter than this cannot be a real annotated event.
HARD_MIN_EVENT_SECONDS = 0.05
EVAL_SPLITS = ("test", "interference")
TRAIN_SPLITS = ("train", "val")


@dataclass
class CheckReport:
    root: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _load_labels(rec_dir: Path) -> dict:
    return json.loads((rec_dir / "labels.json").read_text())


def _check_recording(rec_dir: Path, split: str, expected_rid: str,
                     report: CheckReport, min_event_seconds: float) -> dict:
    labels = _load_labels(rec_dir)
    # Directory ids and labels.json recording_id are intentionally different
    # identifiers (labels must never be inferred from folder names); only the
    # presence/uniqueness of the internal id is enforced at dataset level.
    if not labels.get("recording_id"):
        report.fail(f"{split}/{expected_rid}: labels.json has no recording_id")
    events = labels.get("events", [])
    short_events = 0
    audited = pending = unclaimed = 0
    trainable = 0

    for ev in events:
        eid = ev.get("event_id", "<no-id>")
        start, end = ev.get("start"), ev.get("end")
        if start is None or end is None or end <= start:
            report.fail(f"{expected_rid}/{eid}: invalid span [{start}, {end}]")
            continue
        duration = end - start
        if duration < HARD_MIN_EVENT_SECONDS:
            report.fail(
                f"{expected_rid}/{eid}: span {duration:.3f}s shorter than hard "
                f"floor {HARD_MIN_EVENT_SECONDS}s")
        elif duration < min_event_seconds:
            short_events += 1
            report.warn(
                f"{expected_rid}/{eid}: short event {duration:.3f}s "
                f"(< {min_event_seconds}s)")
        if ev.get("class") not in TARGET_CLASSES + (UNKNOWN_CLASS,):
            report.fail(f"{expected_rid}/{eid}: unknown class {ev.get('class')!r}")
        status = ev.get("audit_status")
        if status not in ("audited", "pending_review", "unclaimed"):
            report.fail(f"{expected_rid}/{eid}: bad audit_status {status!r}")
        if status == AUDIT_AUDITED:
            audited += 1
        elif status == AUDIT_UNCLAIMED:
            unclaimed += 1
        else:
            pending += 1

        source_rel = ev.get("source")
        source_exists = bool(source_rel) and (rec_dir / source_rel).exists()
        if ev.get("class") in TARGET_CLASSES:
            if is_trainable_event(ev):
                trainable += 1
                if not source_exists:
                    report.fail(
                        f"{expected_rid}/{eid}: packed as audited but source "
                        f"track {source_rel} is missing")
                if ev.get("identity_uncertain"):
                    report.fail(
                        f"{expected_rid}/{eid}: trainable event flagged uncertain")
            if status == AUDIT_UNCLAIMED and source_exists:
                report.fail(
                    f"{expected_rid}/{eid}: marked unclaimed but its source "
                    "track is still on disk")

    # Quarantine bookkeeping consistency.
    quarantine = labels.get("quarantine")
    if quarantine:
        moved_ids = {m["event_id"] for m in quarantine.get("moved", [])}
        actual_unclaimed = {
            e["event_id"] for e in events
            if e.get("audit_status") == AUDIT_UNCLAIMED
        }
        if moved_ids != actual_unclaimed:
            report.fail(
                f"{expected_rid}: quarantine record {sorted(moved_ids)} does "
                f"not match unclaimed events {sorted(actual_unclaimed)}")

    # File digests: every declared WAV must exist with an unchanged digest.
    stored = labels.get("file_digests", {})
    if not stored:
        report.fail(f"{expected_rid}: labels.json has no file_digests")
    actual = recording_digests(rec_dir)
    for rel_path, digest in stored.items():
        if rel_path not in actual:
            report.fail(f"{expected_rid}: missing file {rel_path}")
        elif actual[rel_path] != digest:
            report.fail(f"{expected_rid}: digest mismatch for {rel_path}")
    for rel_path in actual:
        if rel_path not in stored:
            report.fail(f"{expected_rid}: undocumented file {rel_path}")

    # Legacy one-to-many mapping coverage.
    legacy = labels.get("legacy_annotations", [])
    coverage = check_legacy_coverage(events, legacy)
    for message in coverage["errors"]:
        report.fail(f"{expected_rid}: {message}")

    return {
        "n_events": len(events),
        "audited": audited,
        "pending_review": pending,
        "unclaimed": unclaimed,
        "trainable": trainable,
        "short_events": short_events,
        "legacy_entries": len(legacy),
        "digests": actual,
        "events_rows": events,
    }


def check_dataset(root: Path | str, audio_cfg=None) -> CheckReport:
    root = Path(root)
    report = CheckReport(root=str(root))
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        report.fail(f"manifest.json missing under {root}")
        return report
    manifest = read_manifest(root)

    # Split relation: whole-recording unit, no train/eval overlap.
    try:
        assert_no_leakage(manifest)
    except AssertionError as exc:
        report.fail(str(exc))
    all_ids = [rid for ids in manifest.values() for rid in ids]
    train_eval_ids = (
        [rid for s in TRAIN_SPLITS for rid in manifest.get(s, [])]
        + [rid for s in EVAL_SPLITS for rid in manifest.get(s, [])]
    )
    if len(train_eval_ids) != len(set(train_eval_ids)):
        report.fail("same recording appears in both training and evaluation splits")

    min_event_seconds = getattr(
        audio_cfg, "min_event_seconds", 0.15) if audio_cfg else 0.15
    # Recreate exactly the analysis-window grid the feature extractor uses for
    # a fixed-duration clip, so the packing check matches training behavior.
    duration_frames = int(12.0 / audio_cfg.frame_seconds) + 1
    window_grid = frame_times(
        np.asarray(window_starts(duration_frames, audio_cfg)), audio_cfg)

    totals = {"recordings": 0, "events": 0, "audited": 0,
              "pending_review": 0, "unclaimed": 0, "trainable": 0,
              "short_events": 0, "legacy_entries": 0}
    per_recording_digests: dict[str, dict] = {}

    for split, ids in manifest.items():
        for rid in ids:
            rec_dir = root / split / rid
            if not rec_dir.exists():
                report.fail(f"{split}/{rid}: recording directory missing")
                continue
            info = _check_recording(
                rec_dir, split, rid, report, min_event_seconds)
            totals["recordings"] += 1
            for key in ("n_events", "audited", "pending_review", "unclaimed",
                        "trainable", "short_events", "legacy_entries"):
                stat_key = "events" if key == "n_events" else key
                totals[stat_key] += info[key]
            per_recording_digests[rid] = info["digests"]

            # Packing rule on training/validation splits: audited-only labels
            # must be a strict subset of full labels. A violation means an
            # audited interval that earns a positive window under full labels
            # loses that window under audited-only packing. Intervals shorter
            # than the minimum window overlap never earn a window either way
            # and are therefore not part of this relation.
            if split in TRAIN_SPLITS:
                full = label_windows(info["events_rows"], window_grid,
                                     audio_cfg, audited_only=False)
                packed = label_windows(info["events_rows"], window_grid,
                                       audio_cfg, audited_only=True)
                if np.any(packed & ~full.astype(bool)):
                    report.fail(
                        f"{rid}: audited packing labels exceed full ground truth")
                full_hits = {
                    e["event_id"] for e in info["events_rows"]
                    if _span_hits_window(
                        e, window_grid, audio_cfg, full.astype(bool))
                }
                packed_hits = {
                    e["event_id"] for e in info["events_rows"]
                    if _span_hits_window(
                        e, window_grid, audio_cfg, packed.astype(bool))
                }
                violated = sorted(
                    (full_hits - packed_hits)
                    & {e["event_id"] for e in info["events_rows"]
                       if is_trainable_event(e)}
                )
                if violated:
                    report.fail(
                        f"{rid}: audited events {violated} were withheld from "
                        "the training pack (packing rule violated)")

    # Repeated material: a source WAV reused across recordings would leak
    # content even though the recording ids differ.
    for message in find_duplicate_material(per_recording_digests):
        report.fail(f"duplicate source material: {message}")

    # Stale material: recordings left on disk from an earlier generation but no
    # longer referenced by the manifest must not silently remain in the data
    # root. In a clean directory this set is empty, so the check never reports
    # a false positive there.
    orphans: list[str] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        registered = set(manifest.get(split_dir.name, []))
        for rec_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if rec_dir.name not in registered:
                orphans.append(f"{split_dir.name}/{rec_dir.name}")
    for orphan in orphans:
        report.fail(f"stale recording not in manifest: {orphan}")

    totals["split_sizes"] = {s: len(ids) for s, ids in manifest.items()}
    totals["n_duplicate_sources"] = len(
        find_duplicate_material(per_recording_digests))
    totals["stale_orphan_dirs"] = len(orphans)
    report.stats = totals
    return report


def _span_hits_window(event: dict, window_grid: np.ndarray,
                      audio_cfg, window_labels: np.ndarray) -> bool:
    """Whether this event's class is positive in a window overlapping it."""
    if event["class"] not in TARGET_CLASSES:
        return False
    cls_idx = TARGET_CLASSES.index(event["class"])
    for i, w_start in enumerate(window_grid):
        overlap = min(w_start + audio_cfg.window_seconds, event["end"]) - max(
            w_start, event["start"])
        if overlap >= audio_cfg.min_label_overlap and window_labels[i, cls_idx]:
            return True
    return False
