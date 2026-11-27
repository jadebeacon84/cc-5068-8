import numpy as np

from ser.cue import CueSheet
from ser.events import (
    SoundEvent,
    _apply_breaks,
    _merge_runs,
    merge_events,
    split_event,
    window_to_frame_prob,
)


def test_runs_merge_small_gap_but_keep_large_gap():
    runs = [(0, 5), (7, 10), (30, 35)]
    merged = _merge_runs(runs, max_gap_frames=3)
    assert merged == [(0, 10), (30, 35)]


def test_silence_breaks_are_applied():
    spans = _apply_breaks(0.0, 4.0, [1.5, 2.8])
    assert spans == [(0.0, 1.5), (1.5, 2.8), (2.8, 4.0)]


def test_assemble_merges_cross_window_event(trained_pipeline, data_dir):
    pipe = trained_pipeline
    pipe.load_recordings()
    rec = next(r for r in pipe._recordings.values() if r.split == "train")
    result = pipe.run_inference(rec.recording_id, force=True)
    for ev in result["events"]:
        # one interval even when supported by multiple overlapping windows
        if len(ev.windows) > 1:
            assert ev.end > ev.start
            assert ev.cls in ("knock", "bell", "chime", "machine")


def test_below_threshold_windows_are_rejected(trained_pipeline):
    pipe = trained_pipeline
    rec = list(pipe._recordings.values())[0]
    result = pipe.run_inference(rec.recording_id, force=True)
    for i, decisions in enumerate(
        result["window_probs"] >= result["thresholds"][None, :]
    ):
        if not decisions.any():
            overlapping = [
                e for e in result["events"]
                if i in e.windows
            ]
            # a fully-rejected window cannot itself seed an event
            assert overlapping == [] or any(
                w != i for e in overlapping for w in e.windows
            )


def test_merge_and_split_primitives():
    events = [
        SoundEvent("ev_0", "bell", 1.0, 1.4, 0.8),
        SoundEvent("ev_1", "bell", 1.45, 2.0, 0.7),
        SoundEvent("ev_2", "knock", 3.0, 3.4, 0.9),
    ]
    merged = merge_events(events, "ev_0", "ev_1")
    assert merged is not None
    assert merged.start == 1.0 and merged.end == 2.0
    assert merge_events(events, "ev_0", "ev_2") is None  # different class

    split = split_event(events, "ev_2", at=3.2)
    assert split is not None
    left, right = split
    assert left.start == 3.0 and right.end == 3.4
    assert left.end < right.start


def test_cut_section_merges_and_restore_rebuilds(tmp_path):
    sheet = CueSheet(tmp_path / "cue.json")
    rid = "rec_x"
    sheet.entries[rid] = [
        {"event_uid": "ev_0", "cls": "bell", "start": 1.0, "end": 1.4,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "ev_1", "cls": "bell", "start": 1.5, "end": 2.0,
         "confidence": 0.8, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "ev_2", "cls": "knock", "start": 4.0, "end": 4.4,
         "confidence": 0.7, "possible_count": 1, "cue_status": "predicted"},
    ]
    before = len(sheet.entries[rid])
    out = sheet.cut_section(rid, 1.38, 1.52)
    assert len(out["entries"]) == before - 1
    merged = next(e for e in out["entries"] if e["cls"] == "bell")
    assert merged["cue_status"] == "merged_by_cut"

    restored = sheet.restore_section(rid)
    assert len(restored["entries"]) == before
    assert sheet.cuts[rid][0]["restored"] is True


def test_frame_probability_covers_all_frames():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    starts = np.array([0, 2], dtype=np.int64)
    frame = window_to_frame_prob(probs, starts, n_frames=7, win_frames=5)
    assert frame.shape == (7, 2)
    assert np.isfinite(frame).all()


def test_cut_geometry_handles_all_positions(tmp_path):
    sheet = CueSheet(tmp_path / "cue_geom.json")
    rid = "r"
    sheet.entries[rid] = [
        {"event_uid": "before", "cls": "knock", "start": 0.2, "end": 0.6,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "after", "cls": "knock", "start": 4.0, "end": 4.5,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "inside", "cls": "machine", "start": 2.1, "end": 2.4,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "left", "cls": "bell", "start": 1.5, "end": 2.2,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "right", "cls": "bell", "start": 2.8, "end": 3.4,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
        {"event_uid": "span", "cls": "machine", "start": 1.8, "end": 3.2,
         "confidence": 0.9, "possible_count": 1, "cue_status": "predicted"},
    ]
    out = sheet.cut_section(rid, 2.0, 3.0)  # 1.0 s removed
    by_id = {e["event_uid"]: e for e in out["entries"]}
    assert by_id["before"]["start"] == 0.2 and by_id["before"]["end"] == 0.6
    assert by_id["after"]["start"] == 3.0 and by_id["after"]["end"] == 3.5
    assert "inside" not in by_id  # fully swallowed marker is dropped
    assert by_id["left"]["end"] == 2.0
    assert by_id["right"]["start"] == 2.0
    assert by_id["right"]["end"] == 2.4
    assert by_id["span"]["start"] == 1.8 and by_id["span"]["end"] == 2.2
    assert all(e["end"] > e["start"] for e in out["entries"])
