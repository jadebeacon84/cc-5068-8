"""Orchestration layer used by both the CLI scripts and the local web UI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    CUESHEET_PATH,
    DATA_ROOT,
    DEFAULT_CONFIG,
    EXPERIMENTS_ROOT,
    TARGET_CLASSES,
    Config,
)
from .cue import CueSheet
from .dataset import (
    assert_no_leakage,
    build_training_arrays,
    load_split,
    read_manifest,
)
from .events import infer_recording
from .evaluation import evaluate_recording, summarize, write_report
from .labeling import AUDIT_STATUSES, set_audit_status
from .model import SoundEventModel, build_metadata, train_models


class Pipeline:
    def __init__(self, config: Config | None = None,
                 data_root: Path = DATA_ROOT,
                 experiments_root: Path = EXPERIMENTS_ROOT,
                 model_dir: Path | None = None):
        self.cfg = config or DEFAULT_CONFIG
        self.data_root = Path(data_root)
        self.experiments_root = Path(experiments_root)
        self.model_dir = Path(model_dir) if model_dir else self.experiments_root / "model"
        self.model: SoundEventModel | None = None
        self._recordings: dict[str, object] = {}
        self._inference: dict[str, dict] = {}
        self.cue_sheet = CueSheet(CUESHEET_PATH)

    # ------------------------------------------------------------ loading
    def require_data(self) -> dict:
        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "dataset not found; run: python3 -m scripts.prepare_data"
            )
        manifest = read_manifest(self.data_root)
        assert_no_leakage(manifest)
        return manifest

    def load_model(self) -> SoundEventModel:
        if self.model is None:
            self.model = SoundEventModel.load(self.model_dir)
        return self.model

    def load_recordings(self, splits=("train", "val", "test", "interference")):
        self.require_data()
        fb = self.model.filterbank if self.model else None
        for split in splits:
            for rec in load_split(self.data_root, split, self.cfg.audio, fb):
                self._recordings[rec.recording_id] = rec
        return self._recordings

    def get_recording(self, recording_id: str):
        if recording_id not in self._recordings:
            self.load_recordings()
        rec = self._recordings.get(recording_id)
        if rec is None:
            raise KeyError(recording_id)
        return rec

    def list_recordings(self) -> list[dict]:
        self.load_recordings()
        out = []
        for rec in self._recordings.values():
            out.append({
                "recording_id": rec.recording_id,
                "split": rec.split,
                "condition": rec.labels.get("condition"),
                "duration": rec.duration,
                "n_events": len(rec.target_events()),
                "n_unknown": len(rec.unknown_events()),
                "path": str(rec.path.relative_to(self.data_root)),
            })
        return sorted(out, key=lambda r: (r["split"], r["recording_id"]))

    # ------------------------------------------------------------ training
    def train(self) -> dict:
        manifest = self.require_data()
        train_recs = load_split(self.data_root, "train", self.cfg.audio)
        val_recs = load_split(self.data_root, "val", self.cfg.audio)
        assert_no_leakage(manifest)
        arrays = build_training_arrays(self.cfg.audio, train_recs, val_recs)
        models, thresholds = train_models(
            arrays["x_train"], arrays["y_train"],
            arrays["x_val"], arrays["y_val"],
            seed=self.cfg.dataset.seed,
        )
        metadata = build_metadata(train_recs, val_recs, arrays["x_train"])
        metadata["thresholds"] = dict(zip(
            list(TARGET_CLASSES), thresholds.round(4).tolist()
        ))
        self.model = SoundEventModel(
            cfg=self.cfg,
            standardizer=arrays["standardizer"],
            models=models,
            thresholds=thresholds,
            metadata=metadata,
        )
        self.model.save(self.model_dir)
        return metadata

    # ----------------------------------------------------------- inference
    def run_inference(self, recording_id: str, force: bool = False) -> dict:
        if recording_id in self._inference and not force:
            return self._inference[recording_id]
        model = self.load_model()
        rec = self.get_recording(recording_id)
        result = infer_recording(model, rec)
        result["annotations"] = rec.labels.get("events", [])
        result["legacy_annotations"] = rec.labels.get("legacy_annotations", [])
        self._inference[recording_id] = result
        if recording_id not in self.cue_sheet.entries:
            self.cue_sheet.replace_from_events(recording_id, result["events"])
        return result

    # ------------------------------------------------------------ auditing
    def audit_event(self, recording_id: str, event_id: str,
                    status: str, note: str | None = None) -> dict:
        """Record the listening-review verdict for one annotated interval.

        The change is written to labels.json and immediately constrains
        training packing: only ``audited`` intervals keep contributing
        positive windows; pending/unclaimed intervals are withheld.
        """
        if status not in AUDIT_STATUSES:
            raise ValueError(f"unknown audit status {status!r}")
        rec = self.get_recording(recording_id)
        labels_path = rec.path / "labels.json"
        updated = set_audit_status(rec.labels["events"], event_id, status, note)
        labels_path.write_text(json.dumps(rec.labels, indent=2, ensure_ascii=False))
        self._inference.pop(recording_id, None)
        return updated

    def annotation_summary(self, recording_id: str) -> dict:
        rec = self.get_recording(recording_id)
        from .labeling import events_by_audit, trainable_events

        events = rec.labels.get("events", [])
        buckets = events_by_audit(events)
        return {
            "recording_id": recording_id,
            "audit_counts": {k: len(v) for k, v in buckets.items()},
            "trainable_event_ids": [e["event_id"] for e in trainable_events(events)],
            "legacy_annotations": rec.labels.get("legacy_annotations", []),
        }

    def run_split_evaluation(self, split: str) -> dict:
        model = self.load_model()
        recs = load_split(self.data_root, split, self.cfg.audio, model.filterbank)
        per_recording = []
        for rec in recs:
            result = infer_recording(model, rec)
            report = evaluate_recording(result["events"], rec.labels)
            per_recording.append(report)
        summary = summarize(per_recording)
        out_dir = self.experiments_root / "evaluation"
        write_report(out_dir / f"{split}_report.json", per_recording, summary,
                     extra={"split": split, "iou_threshold": 0.3})
        return summary

    # ------------------------------------------------------------- evidence
    def evidence_payload(self, recording_id: str, window_index: int) -> dict:
        result = self.run_inference(recording_id)
        rec = self.get_recording(recording_id)
        idx = int(window_index)
        if not (0 <= idx < len(result["window_starts"])):
            raise IndexError("window index out of range")
        start_frame = int(result["window_starts"][idx])
        win = self.cfg.audio.win_frames
        patch = result["spectrogram"][:, start_frame:start_frame + win]
        t0 = start_frame * self.cfg.audio.frame_seconds
        t1 = t0 + self.cfg.audio.window_seconds
        overlapping_events = [
            e.to_dict() for e in result["events"]
            if e.end > t0 and e.start < t1
        ]
        gt_in_window = [
            e for e in rec.labels["events"]
            if e["end"] > t0 and e["start"] < t1
        ]
        return {
            "recording_id": recording_id,
            "window_index": idx,
            "window_start": round(t0, 3),
            "window_end": round(t1, 3),
            "sample_rate": self.cfg.audio.sample_rate,
            "frame_seconds": self.cfg.audio.frame_seconds,
            "n_mels": self.cfg.audio.n_mels,
            "mel_patch": np.asarray(patch, dtype=np.float32).tolist(),
            "class_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(TARGET_CLASSES, result["window_probs"][idx])
            },
            "thresholds": {
                cls: round(float(t), 4)
                for cls, t in zip(TARGET_CLASSES, result["thresholds"])
            },
            "decisions": {
                cls: bool(p >= t)
                for cls, p, t in zip(
                    TARGET_CLASSES,
                    result["window_probs"][idx],
                    result["thresholds"],
                )
            },
            "predicted_events": overlapping_events,
            "ground_truth_in_window": gt_in_window,
        }

    def waveform_payload(self, recording_id: str, bins: int = 1200) -> dict:
        rec = self.get_recording(recording_id)
        audio = rec.audio
        trimmed = len(audio) - (len(audio) % bins)
        reshaped = audio[:trimmed].reshape(bins, -1)
        peaks = np.max(np.abs(reshaped), axis=1)
        return {
            "recording_id": recording_id,
            "sample_rate": rec.sample_rate,
            "duration": rec.duration,
            "peaks": peaks.astype(np.float32).tolist(),
        }
