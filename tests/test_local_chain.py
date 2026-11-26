"""Guardrails: no remote recognition dependency, no folder-name classification."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

FORBIDDEN_IMPORTS = (
    "import requests",
    "from requests",
    "import http.client",
    "from http.client",
    "import urllib.request",
    "from urllib.request",
    "import aiohttp",
    "from aiohttp",
)


def test_processing_chain_has_no_remote_calls():
    pkg = Path(__file__).resolve().parents[1] / "ser"
    offenders = []
    for py in pkg.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in FORBIDDEN_IMPORTS:
            if token in text:
                offenders.append(f"{py.name}: {token}")
    assert not offenders, f"remote-call imports found: {offenders}"


def test_predictions_independent_of_folder_name(trained_pipeline, tmp_path):
    pipe = trained_pipeline
    rec = list(pipe._recordings.values())[0]
    base = pipe.run_inference(rec.recording_id, force=True)
    base_probs = base["window_probs"].copy()

    # Copy the recording to an unrelated, misleadingly named directory and run
    # the same local chain; probabilities must be identical because labels
    # only come from labels.json + audio content.
    decoys = [
        tmp_path / "train" / "bell_bell_bell",
        tmp_path / "test" / "totally_knock_folder",
    ]
    for target in decoys:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(rec.path, target)
        from ser.dataset import load_recording, recording_patches

        new_rec = load_recording(target, "test", pipe.cfg.audio, pipe.model.filterbank)
        result = pipe.model.score_windows(
            recording_patches(new_rec, pipe.cfg.audio.win_frames)
        )
        assert np.allclose(result, base_probs, atol=1e-6), (
            "prediction changed after renaming the folder"
        )
        shutil.rmtree(target.parent)


def test_background_is_not_equated_with_target(trained_pipeline):
    pipe = trained_pipeline
    rec = list(pipe._recordings.values())[0]
    result = pipe.run_inference(rec.recording_id, force=True)
    decisions = result["window_probs"] >= result["thresholds"][None, :]
    # Rejected windows are simply dropped: they never produce an event label.
    for i, keep in enumerate(decisions.any(axis=1)):
        if not keep:
            assert not any(i in e.windows and len(e.windows) == 1
                           for e in result["events"])
    for ev in result["events"]:
        assert ev.cls in ("knock", "bell", "machine")
        assert ev.source_match in ("exact", "count_only", "none")


def test_background_only_recording_rejects_everything(trained_pipeline):
    pipe = trained_pipeline
    pipe.load_recordings()
    bg = [r for r in pipe._recordings.values()
          if r.labels.get("background_only")]
    assert bg, "synthesis must include pure-background recordings"
    for rec in bg:
        result = pipe.run_inference(rec.recording_id, force=True)
        # A pure background/noise recording must not generate target events.
        assert result["events"] == [], (
            f"background recording {rec.recording_id} produced "
            f"{len(result['events'])} false events"
        )
