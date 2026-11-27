import json
from pathlib import Path

from ser.audio import load_wav
from ser.config import TARGET_CLASSES, UNKNOWN_CLASS
from ser.dataset import (
    assert_no_leakage,
    label_windows,
    load_split,
    read_manifest,
    recording_patches,
)
from ser.features import mel_filterbank


def test_manifest_has_recording_level_splits(data_dir):
    manifest = read_manifest(data_dir)
    assert set(manifest) == {"train", "val", "test", "interference"}
    assert_no_leakage(manifest)
    all_ids = [rid for ids in manifest.values() for rid in ids]
    assert len(all_ids) == len(set(all_ids))


def test_leakage_guard_detects_shared_recording():
    bad = {"train": ["rec_a", "rec_b"], "test": ["rec_b"]}
    try:
        assert_no_leakage(bad)
    except AssertionError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("guard must fail when a recording crosses splits")


def test_recording_contents_and_source_tracks(data_dir, tiny_config):
    manifest = read_manifest(data_dir)
    rid = manifest["train"][0]
    rec_dir = data_dir / "train" / rid
    labels = json.loads((rec_dir / "labels.json").read_text())
    assert labels["sample_rate"] == tiny_config.audio.sample_rate
    audio, sr = load_wav(rec_dir / "mix.wav")
    assert sr == tiny_config.audio.sample_rate
    assert len(audio) > 0
    classes = {e["class"] for e in labels["events"]}
    assert classes.issubset(set(TARGET_CLASSES) | {UNKNOWN_CLASS})
    for ev in labels["events"]:
        assert ev["end"] > ev["start"]
        assert (rec_dir / ev["source"]).exists()
        # source file names must not encode the class
        assert ev["class"] not in Path(ev["source"]).stem
        assert ev["audit_status"] in (
            "audited", "pending_review", "unclaimed")
    assert "file_digests" in labels and labels["file_digests"]


def test_legacy_bell_split_produces_overlapping_bell_and_chime(data_dir):
    """Two time-overlapping examples share the old bell label across splits."""
    found_classes = set()
    overlap_recordings = 0
    pending_seen = False
    for split in ("train", "val"):
        manifest = read_manifest(data_dir)
        for rid in manifest[split]:
            labels = json.loads(
                (data_dir / split / rid / "labels.json").read_text())
            legacy = labels.get("legacy_annotations", [])
            if not legacy:
                continue
            scopes = {entry["scope"] for entry in legacy}
            assert {"single_event", "whole_segment"}.issubset(scopes)
            bells = [e for e in labels["events"] if e["class"] == "bell"]
            chimes = [e for e in labels["events"] if e["class"] == "chime"]
            assert bells and chimes
            overlap = any(
                min(b["end"], c["end"]) - max(b["start"], c["start"]) > 0
                for b in bells for c in chimes)
            assert overlap, "bell and chime must overlap in time"
            found_classes |= {"bell", "chime"}
            overlap_recordings += 1
            if any(e.get("audit_status") == "pending_review"
                   for e in labels["events"]):
                pending_seen = True
    assert overlap_recordings >= 2
    assert found_classes == {"bell", "chime"}
    assert pending_seen, "one legacy split must remain pending listening review"


def test_chime_class_present_in_taxonomy_and_data(data_dir):
    all_classes = set()
    for split, ids in read_manifest(data_dir).items():
        for rid in ids:
            labels = json.loads(
                (data_dir / split / rid / "labels.json").read_text())
            all_classes |= {e["class"] for e in labels["events"]}
    assert "chime" in TARGET_CLASSES
    assert "chime" in all_classes


def test_window_labels_and_patch_shapes(data_dir, tiny_config):
    recs = load_split(data_dir, "train", tiny_config.audio,
                      mel_filterbank(tiny_config.audio))
    rec = recs[0]
    expected_dim = tiny_config.audio.n_mels * tiny_config.audio.win_frames
    patches = recording_patches(rec, tiny_config.audio.win_frames)
    assert patches.shape == (len(rec.window_times), expected_dim)
    assert rec.window_labels.shape == (len(rec.window_times), len(TARGET_CLASSES))
    # unknown/background windows stay all-zero: nothing defaults to a target
    unknown_events = [e for e in rec.labels["events"]
                      if e["class"] == UNKNOWN_CLASS]
    if unknown_events:
        labels = label_windows(unknown_events, rec.window_times, tiny_config.audio)
        assert labels.sum() == 0


def test_splits_are_disjoint_recordings(data_dir, tiny_config):
    seen = {}
    for split in ("train", "val", "test", "interference"):
        recs = load_split(data_dir, split, tiny_config.audio)
        for rec in recs:
            assert rec.recording_id not in seen
            seen[rec.recording_id] = split
