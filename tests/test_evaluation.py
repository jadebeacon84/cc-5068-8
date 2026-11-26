from ser.config import TARGET_CLASSES
from ser.evaluation import evaluate_recording, match_events, summarize
from ser.events import SoundEvent


def _gt():
    return [
        {"event_id": "g1", "class": "bell", "start": 1.0, "end": 2.0,
         "source": "sources/g1.wav"},
        {"event_id": "g2", "class": "knock", "start": 4.0, "end": 4.3,
         "source": "sources/g2.wav"},
        {"event_id": "g3", "class": "unknown", "start": 6.0, "end": 7.0,
         "source": "sources/g3.wav"},
    ]


def test_event_level_match_and_miss():
    pred = [
        SoundEvent("ev_0", "bell", 1.05, 1.95, 0.9),
        SoundEvent("ev_1", "machine", 8.0, 8.6, 0.6),
    ]
    result = match_events(pred, _gt())
    assert len(result["tp"]) == 1
    assert result["tp"][0]["gt_event_id"] == "g1"
    assert any(f["event"]["cls"] == "machine" for f in result["fp"])
    assert any(f["class"] == "knock" for f in result["fn"])


def test_unknown_rejection_and_background_false_alarm():
    # prediction on the unknown interval must be flagged; isolated prediction
    # with no ground-truth overlap is a background false alarm
    pred = [
        SoundEvent("ev_0", "bell", 6.1, 6.8, 0.4),
        SoundEvent("ev_1", "machine", 9.0, 9.8, 0.4),
    ]
    report = evaluate_recording(pred, {"recording_id": "r", "events": _gt()})
    reasons = {f["reason"] for f in report["fp"]}
    assert "unknown_sound_not_rejected" in reasons
    assert "background_false_alarm" in reasons
    assert report["unknown_total"] == 1 and report["unknown_rejected"] == 0


def test_clean_rejection_scores_one():
    pred = [SoundEvent("ev_0", "bell", 1.02, 1.98, 0.9)]
    report = evaluate_recording(pred, {"recording_id": "r", "events": _gt()})
    summary = summarize([report])
    assert summary["unknown_rejection"]["rejection_rate"] == 1.0
    assert summary["event_level"]["tp"] == 1
    assert summary["event_level"]["fn"] == 1  # the knock was never predicted


def test_pipeline_evaluation_on_held_out(trained_pipeline):
    pipe = trained_pipeline
    summary = pipe.run_split_evaluation("test")
    assert summary["event_level"]["tp"] >= 1
    for cls in TARGET_CLASSES:
        assert cls in summary["per_class"]
    assert summary["false_alarms"]["total"] >= 0
    assert summary["unknown_rejection"]["total"] >= 1
