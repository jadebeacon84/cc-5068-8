"""Label taxonomy maintenance: mapping, audit gating, quarantine, checks.

These tests operate on freshly generated clean directories, never on
hand-edited data, and every induced failure is created inside a tmp_path copy.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ser.config import (
    AUDIT_AUDITED,
    AUDIT_PENDING,
    AUDIT_UNCLAIMED,
    SCOPE_SEGMENT,
    SCOPE_SINGLE,
    TARGET_CLASSES,
)
from ser.dataset import label_windows
from ser.features import frame_times, window_starts
from ser.label_check import check_dataset
from ser.labeling import (
    CURRENT_TO_LEGACY,
    LEGACY_TO_CURRENT,
    check_legacy_coverage,
    events_by_audit,
    is_trainable_event,
    quarantine_source_tracks,
    resolve_legacy_label,
    restore_source_tracks,
    set_audit_status,
)
from ser.release import build_release, verify_quarantine_cycle


# ------------------------------------------------------------- legacy mapping
def test_legacy_bell_is_one_to_many_and_reverse_maps():
    assert LEGACY_TO_CURRENT["bell"] == ("bell", "chime")
    assert resolve_legacy_label("bell").resolved is False
    decided = resolve_legacy_label("bell", "chime")
    assert decided.resolved and decided.candidates == ("bell", "chime")
    assert resolve_legacy_label("knock").candidates == ("knock",)
    assert CURRENT_TO_LEGACY["chime"] == "bell"
    assert CURRENT_TO_LEGACY["bell"] == "bell"


def test_legacy_coverage_single_and_segment_scopes():
    events = [
        {"event_id": "a", "class": "bell", "start": 1.0, "end": 2.0,
         "audit_status": AUDIT_AUDITED},
        {"event_id": "b", "class": "chime", "start": 1.6, "end": 2.4,
         "audit_status": AUDIT_AUDITED},
    ]
    legacy = [
        {"legacy_id": "L1", "legacy_class": "bell", "start": 1.0, "end": 2.0,
         "scope": SCOPE_SINGLE},
        {"legacy_id": "L2", "legacy_class": "bell", "start": 1.0, "end": 2.4,
         "scope": SCOPE_SEGMENT},
    ]
    report = check_legacy_coverage(events, legacy)
    assert report["single_event"][0]["ok"] is True
    assert report["whole_segment"][0]["resolved"] is True
    assert report["errors"] == []

    # A wrong-class event under a legacy span must be reported, not silently
    # reassigned to one of the split classes.
    bad = [{"event_id": "c", "class": "machine", "start": 3.0, "end": 3.4,
            "audit_status": AUDIT_AUDITED}]
    bad_legacy = [{"legacy_id": "L3", "legacy_class": "bell",
                   "start": 3.0, "end": 3.4, "scope": SCOPE_SINGLE}]
    bad_report = check_legacy_coverage(bad, bad_legacy)
    assert bad_report["errors"]


def test_unresolved_split_segment_is_flagged():
    events = [
        {"event_id": "a", "class": "bell", "start": 1.0, "end": 2.0,
         "audit_status": AUDIT_AUDITED, "identity_uncertain": False},
        {"event_id": "b", "class": "chime", "start": 1.6, "end": 2.2,
         "audit_status": AUDIT_PENDING, "identity_uncertain": True},
    ]
    legacy = [{"legacy_id": "L", "legacy_class": "bell",
               "start": 1.0, "end": 2.2, "scope": SCOPE_SEGMENT}]
    report = check_legacy_coverage(events, legacy)
    segment = report["whole_segment"][0]
    assert segment["resolved"] is False
    assert "b" in segment["unresolved_event_ids"]


# ------------------------------------------------------------- audit gating
def _window_grid(audio_cfg):
    n_frames = int(12.0 / audio_cfg.frame_seconds) + 1
    return frame_times(np.asarray(window_starts(n_frames, audio_cfg)),
                       audio_cfg)


def test_only_audited_intervals_enter_training_windows(data_dir, tiny_config):
    rec_dir = _find_recording_with_pending(data_dir)
    labels = json.loads((rec_dir / "labels.json").read_text())
    grid = _window_grid(tiny_config.audio)
    full = label_windows(labels["events"], grid, tiny_config.audio,
                         audited_only=False)
    packed = label_windows(labels["events"], grid, tiny_config.audio,
                           audited_only=True)
    # Audited packing never adds positives beyond the full ground truth.
    assert not np.any(packed & ~full.astype(bool))
    pending = [e for e in labels["events"]
               if e.get("audit_status") == AUDIT_PENDING]
    assert pending, "synthesis must include a pending listening-review item"
    for ev in pending:
        assert not is_trainable_event(ev)
        idx = TARGET_CLASSES.index(ev["class"])
        contributing = [
            i for i, t in enumerate(grid)
            if min(t + tiny_config.audio.window_seconds, ev["end"])
            - max(t, ev["start"]) >= tiny_config.audio.min_label_overlap
        ]
        assert all(packed[i, idx] == 0 for i in contributing)


def test_audit_verdict_changes_packability():
    event = {"event_id": "x", "class": "chime", "start": 1.0, "end": 1.5,
             "source": "sources/x.wav", "audit_status": AUDIT_PENDING,
             "identity_uncertain": True}
    assert not is_trainable_event(event)
    set_audit_status([event], "x", AUDIT_AUDITED, note="review ok")
    assert is_trainable_event(event) and event["identity_uncertain"] is False
    buckets = events_by_audit([event])
    assert len(buckets[AUDIT_AUDITED]) == 1


# ------------------------------------------------------------- quarantine
def test_quarantine_downgrades_and_restore_recovers_identity(
        data_dir, tiny_config, tmp_path):
    rec_dir = _single_recording_case(data_dir, tmp_path / "quarantine")
    labels = json.loads((rec_dir / "labels.json").read_text())
    target = next(e for e in labels["events"]
                  if e["class"] in TARGET_CLASSES and is_trainable_event(e))
    original = dict(target)

    result = quarantine_source_tracks(rec_dir, [target["event_id"]])
    moved = next(m for m in result["moved"]
                 if m["event_id"] == target["event_id"])
    assert moved["previous_audit_status"] == AUDIT_AUDITED
    after = json.loads((rec_dir / "labels.json").read_text())
    ev = next(e for e in after["events"]
              if e["event_id"] == target["event_id"])
    assert ev["audit_status"] == AUDIT_UNCLAIMED
    assert not (rec_dir / original["source"]).exists()
    assert not is_trainable_event(ev)

    restored = restore_source_tracks(rec_dir)
    record = next(r for r in restored["restored"]
                  if r["event_id"] == target["event_id"])
    final = json.loads((rec_dir / "labels.json").read_text())
    final_ev = next(e for e in final["events"]
                    if e["event_id"] == target["event_id"])
    assert record["audit_status"] == AUDIT_AUDITED
    assert (rec_dir / original["source"]).exists()
    assert final_ev["class"] == original["class"]
    assert (final_ev["start"], final_ev["end"]) == (
        original["start"], original["end"])
    assert is_trainable_event(final_ev)


def test_quarantine_cycle_verifies_by_interval_id(
        data_dir, tiny_config, tmp_path):
    rec_dir = _single_recording_case(data_dir, tmp_path / "cycle")
    labels = json.loads((rec_dir / "labels.json").read_text())
    event_id = next(
        e["event_id"] for e in labels["events"] if is_trainable_event(e))
    outcome = verify_quarantine_cycle(
        rec_dir, event_id, tiny_config.audio, "sources_quarantined")
    assert outcome["checks"]["ok"] is True
    assert outcome["checks"]["event_id_preserved"] is True


# ------------------------------------------------------------- label checks
def test_clean_dataset_passes_without_false_alarms(data_dir, tiny_config):
    report = check_dataset(data_dir, tiny_config.audio)
    assert report.ok, report.errors
    assert report.stats["pending_review"] >= 1
    assert report.stats["legacy_entries"] >= 2
    assert report.stats["n_duplicate_sources"] == 0
    assert report.stats["stale_orphan_dirs"] == 0


def test_check_flags_stale_directories_outside_manifest(
        data_dir, tiny_config, tmp_path):
    root = tmp_path / "stale"
    shutil.copytree(data_dir, root)
    manifest = json.loads((root / "manifest.json").read_text())
    # Simulate a recording left behind by an earlier generation.
    old = root / "train" / manifest["train"][0]
    stale = root / "train" / "tr_stale00000"
    shutil.copytree(old, stale)
    report = check_dataset(root, tiny_config.audio)
    assert not report.ok
    assert any("stale recording not in manifest" in e for e in report.errors)
    assert report.stats["stale_orphan_dirs"] == 1


def test_check_detects_bad_duration_and_missing_file(
        data_dir, tiny_config, tmp_path):
    rec_dir = _single_recording_case(data_dir, tmp_path / "duration")
    labels_path = rec_dir / "labels.json"
    labels = json.loads(labels_path.read_text())
    labels["events"][0]["end"] = labels["events"][0]["start"]  # zero duration
    labels_path.write_text(json.dumps(labels))
    report = check_dataset(rec_dir.parents[1], tiny_config.audio)
    assert any("invalid span" in e for e in report.errors)

    rec_dir2 = _single_recording_case(data_dir, tmp_path / "missing")
    labels2 = json.loads((rec_dir2 / "labels.json").read_text())
    (rec_dir2 / labels2["events"][0]["source"]).unlink()
    report2 = check_dataset(rec_dir2.parents[1], tiny_config.audio)
    assert any("missing file" in e for e in report2.errors)


def test_check_flags_short_event_as_warning(
        data_dir, tiny_config, tmp_path):
    rec_dir = _single_recording_case(data_dir, tmp_path / "short")
    labels_path = rec_dir / "labels.json"
    labels = json.loads(labels_path.read_text())
    # 0.10s is below the 0.15s configured minimum but above the 0.05s hard
    # floor: it must be a warning, not a hard error (no false positives).
    labels["events"][0]["start"] = 1.0
    labels["events"][0]["end"] = 1.10
    labels_path.write_text(json.dumps(labels))
    report = check_dataset(rec_dir.parents[1], tiny_config.audio)
    assert any("short event" in w for w in report.warnings)
    assert report.stats["short_events"] >= 1


def test_check_detects_duplicate_material_and_digest_change(
        data_dir, tiny_config, tmp_path):
    root = tmp_path / "dup"
    shutil.copytree(data_dir, root)
    manifest = json.loads((root / "manifest.json").read_text())
    a_dir = root / "train" / manifest["train"][0]
    b_dir = root / "train" / manifest["train"][1]
    a_labels = json.loads((a_dir / "labels.json").read_text())
    b_labels = json.loads((b_dir / "labels.json").read_text())
    a_src = a_labels["events"][0]["source"]
    b_src_name = Path(b_labels["events"][0]["source"]).name
    shutil.copyfile(a_dir / a_src, b_dir / "sources" / b_src_name)
    # Align the copied digest so only cross-recording duplication is detected.
    b_labels["file_digests"][f"sources/{b_src_name}"] = \
        a_labels["file_digests"][a_src]
    (b_dir / "labels.json").write_text(json.dumps(b_labels))
    report = check_dataset(root, tiny_config.audio)
    assert any("duplicate source material" in e for e in report.errors)

    # Tampering one WAV without updating digests must be detected.
    case_dir = _single_recording_case(data_dir, tmp_path / "tamper")
    t_labels = json.loads((case_dir / "labels.json").read_text())
    wav = case_dir / t_labels["events"][0]["source"]
    raw = wav.read_bytes()
    wav.write_bytes(raw[:-200] + b"\x00" * 200)
    tamper_report = check_dataset(case_dir.parents[1], tiny_config.audio)
    assert any("digest mismatch" in e for e in tamper_report.errors)


def test_check_enforces_train_eval_split_disjoint():
    from ser.dataset import assert_no_leakage

    bad = {"train": ["rec_a"], "val": [], "test": ["rec_a"],
           "interference": []}
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_release_rebuild_passes_clean_checks(
        data_dir, tiny_config, tmp_path):
    release = tmp_path / "release"
    summary = build_release(data_dir, release, tiny_config.audio)
    assert summary["all_cycles_ok"] is True
    assert summary["final_check"]["ok"] is True
    assert (release / "release_report.json").exists()
    # Working dataset is untouched: no quarantine directory leaked into it.
    leaked = [p for s in ("train", "val", "test", "interference")
              for p in (data_dir / s).rglob("sources_quarantined")]
    assert leaked == []


# ------------------------------------------------------------- helpers
def _find_recording_with_pending(data_dir: Path) -> Path:
    for split in ("train", "val", "test", "interference"):
        for rec_dir in sorted((data_dir / split).iterdir()):
            labels = json.loads((rec_dir / "labels.json").read_text())
            if any(e.get("audit_status") == AUDIT_PENDING
                   for e in labels["events"]):
                return rec_dir
    raise AssertionError("no pending-review recording synthesized")


def _single_recording_case(data_dir: Path, case_root: Path) -> Path:
    """Materialize a valid one-split dataset layout under a clean tmp dir."""
    if case_root.exists():
        shutil.rmtree(case_root)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    rid = manifest["train"][0]
    dest = case_root / "train" / rid
    shutil.copytree(data_dir / "train" / rid, dest)
    (case_root / "manifest.json").write_text(
        json.dumps({"train": [rid], "val": [], "test": [],
                    "interference": []}))
    return dest
