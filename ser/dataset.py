"""Dataset assembly with strict recording-level split enforcement.

The split unit is the whole recording. Frame/window slices of one recording are
never spread across train and test, and `assert_no_leakage` fails loudly if a
manifest ever violates that invariant. Labels come from labels.json only —
never from directory or file names.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_wav
from .config import AudioConfig, TARGET_CLASSES, UNKNOWN_CLASS
from .features import Standardizer, extract_windows, frame_times, log_mel_spectrogram, mel_filterbank


@dataclass
class Recording:
    recording_id: str
    split: str
    path: Path
    labels: dict
    audio: np.ndarray
    sample_rate: int
    spectrogram: np.ndarray
    window_starts: np.ndarray
    window_times: np.ndarray
    window_labels: np.ndarray  # (n_windows, n_targets) multi-label, unknown/background => all zeros
    filterbank: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def target_events(self) -> list[dict]:
        return [e for e in self.labels["events"] if e["class"] in TARGET_CLASSES]

    def unknown_events(self) -> list[dict]:
        return [e for e in self.labels["events"] if e["class"] == UNKNOWN_CLASS]


def read_manifest(data_root: Path) -> dict[str, list[str]]:
    return json.loads((Path(data_root) / "manifest.json").read_text())


def assert_no_leakage(manifest: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split, ids in manifest.items():
        if len(ids) != len(set(ids)):
            raise AssertionError(f"duplicate recording ids inside split {split}")
        for rid in ids:
            if rid in seen:
                raise AssertionError(
                    f"leakage: recording {rid} appears in both {seen[rid]} and {split}; "
                    "slices of one recording must not cross splits"
                )
            seen[rid] = split


def label_windows(events: list[dict], window_times: np.ndarray,
                  cfg: AudioConfig) -> np.ndarray:
    """Multi-label each analysis window from ground-truth event spans.

    A window is positive for a class when a matching event overlaps it by at
    least `min_label_overlap` seconds. Unknown and background-only windows stay
    all-zero so the classifier can learn to reject them.
    """
    labels = np.zeros((len(window_times), len(TARGET_CLASSES)), dtype=np.int8)
    for ev in events:
        if ev["class"] not in TARGET_CLASSES:
            continue
        cls_idx = TARGET_CLASSES.index(ev["class"])
        for i, w_start in enumerate(window_times):
            w_end = w_start + cfg.window_seconds
            overlap = min(w_end, ev["end"]) - max(w_start, ev["start"])
            if overlap >= cfg.min_label_overlap:
                labels[i, cls_idx] = 1
    return labels


def load_recording(rec_dir: Path, split: str, cfg: AudioConfig,
                   filterbank: np.ndarray | None = None) -> Recording:
    rec_dir = Path(rec_dir)
    labels = json.loads((rec_dir / "labels.json").read_text())
    audio, sr = load_wav(rec_dir / "mix.wav", target_sr=cfg.sample_rate)
    fb = filterbank if filterbank is not None else mel_filterbank(cfg)
    spec = log_mel_spectrogram(audio, cfg, fb)
    patches, starts = extract_windows(spec, cfg)
    times = frame_times(starts, cfg)
    win_labels = label_windows(labels["events"], times, cfg)
    return Recording(
        recording_id=labels["recording_id"],
        split=split,
        path=rec_dir,
        labels=labels,
        audio=audio,
        sample_rate=sr,
        spectrogram=spec,
        window_starts=starts,
        window_times=times,
        window_labels=win_labels,
        filterbank=fb,
    )


def load_split(data_root: Path, split: str, cfg: AudioConfig,
               filterbank: np.ndarray | None = None) -> list[Recording]:
    data_root = Path(data_root)
    manifest = read_manifest(data_root)
    assert_no_leakage(manifest)
    fb = filterbank if filterbank is not None else mel_filterbank(cfg)
    out = []
    for rid in manifest[split]:
        out.append(load_recording(data_root / split / rid, split, cfg, fb))
    return out


def recording_patches(rec: Recording, win_frames: int) -> np.ndarray:
    n_mels, _ = rec.spectrogram.shape
    rows = [
        rec.spectrogram[:, s:s + win_frames].reshape(-1)
        for s in rec.window_starts
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def collect_window_matrix(recordings: list[Recording],
                          win_frames: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([recording_patches(r, win_frames) for r in recordings], axis=0)
    y = np.concatenate([r.window_labels for r in recordings], axis=0)
    return x, y


def build_training_arrays(cfg: AudioConfig,
                          train_recs: list[Recording],
                          val_recs: list[Recording]) -> dict:
    x_train, y_train = collect_window_matrix(train_recs, cfg.win_frames)
    x_val, y_val = collect_window_matrix(val_recs, cfg.win_frames)
    standardizer = Standardizer.fit(x_train)
    return {
        "x_train": standardizer.transform(x_train),
        "y_train": y_train,
        "x_val": standardizer.transform(x_val),
        "y_val": y_val,
        "standardizer": standardizer,
    }
