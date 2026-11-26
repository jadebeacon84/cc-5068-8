"""Local multi-label classifier and its serialized artifact.

One RandomForest binary classifier is trained per target class (one-vs-rest).
Unknown and background windows carry the all-zero vector, so low-confidence
windows are rejected instead of being forced into a target class. Detection
thresholds are calibrated on validation windows and stored with the model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import Config, TARGET_CLASSES, load_config
from .features import Standardizer, mel_filterbank


def _precision_recall_f1(probs: np.ndarray, labels: np.ndarray,
                         threshold: float) -> tuple[float, float, float]:
    pred = probs >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def calibrate_thresholds(probs: np.ndarray, labels: np.ndarray,
                         default: float = 0.5,
                         precision_floor: float = 0.85) -> np.ndarray:
    """Pick the F1-best threshold subject to a precision floor.

    Unknown and background windows are all-zero negatives; the precision floor
    prevents those from being absorbed into a target class for the sake of
    recall. If no threshold reaches the floor, take the highest-precision one.
    """
    thresholds = np.full(labels.shape[1], default, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 91)
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        scored = [
            (t, *_precision_recall_f1(probs[:, c], labels[:, c], t))
            for t in grid
        ]
        feasible = [row for row in scored if row[1] >= precision_floor]
        if feasible:
            best = max(feasible, key=lambda row: row[3])
        else:
            best = max(scored, key=lambda row: (row[1], row[3]))
        thresholds[c] = best[0]
    return thresholds


def train_models(x_train: np.ndarray, y_train: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 seed: int) -> tuple[dict[str, RandomForestClassifier], np.ndarray]:
    models: dict[str, RandomForestClassifier] = {}
    val_probs = np.zeros_like(y_val, dtype=np.float32)
    for c, cls in enumerate(TARGET_CLASSES):
        rf = RandomForestClassifier(
            n_estimators=240,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed + c,
            class_weight="balanced",
        )
        rf.fit(x_train, y_train[:, c])
        models[cls] = rf
        val_probs[:, c] = rf.predict_proba(x_val)[:, 1]
    thresholds = calibrate_thresholds(val_probs, y_val)
    return models, thresholds


class SoundEventModel:
    """Serializable bundle: filterbank, standardizer, per-class forests."""

    def __init__(self, cfg: Config, standardizer: Standardizer,
                 models: dict[str, RandomForestClassifier],
                 thresholds: np.ndarray, metadata: dict):
        self.cfg = cfg
        self.standardizer = standardizer
        self.models = models
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.metadata = metadata
        self.filterbank = mel_filterbank(cfg.audio)

    def score_windows(self, patches: np.ndarray) -> np.ndarray:
        x = self.standardizer.transform(patches)
        probs = np.zeros((len(x), len(TARGET_CLASSES)), dtype=np.float32)
        for c, cls in enumerate(TARGET_CLASSES):
            probs[:, c] = self.models[cls].predict_proba(x)[:, 1]
        return probs

    def binary_decisions(self, probs: np.ndarray) -> np.ndarray:
        return probs >= self.thresholds[np.newaxis, :]

    def save(self, model_dir: Path) -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "standardizer": self.standardizer,
                "thresholds": self.thresholds,
                "metadata": self.metadata,
            },
            model_dir / "model.joblib",
        )
        np.save(model_dir / "filterbank.npy", self.filterbank)
        self.cfg.save(model_dir / "config_snapshot.json")
        (model_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False)
        )
        return model_dir

    @classmethod
    def load(cls, model_dir: Path) -> "SoundEventModel":
        model_dir = Path(model_dir)
        bundle = joblib.load(model_dir / "model.joblib")
        cfg = load_config(model_dir / "config_snapshot.json")
        obj = cls(
            cfg=cfg,
            standardizer=bundle["standardizer"],
            models=bundle["models"],
            thresholds=bundle["thresholds"],
            metadata=bundle["metadata"],
        )
        obj.filterbank = np.load(model_dir / "filterbank.npy")
        return obj


def build_metadata(train_recs, val_recs, x_train: np.ndarray) -> dict:
    return {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_classes": list(TARGET_CLASSES),
        "feature_dim": int(x_train.shape[1]),
        "train_recordings": [r.recording_id for r in train_recs],
        "val_recordings": [r.recording_id for r in val_recs],
        "train_windows": int(x_train.shape[0]),
        "model_type": "one-vs-rest RandomForest on log-mel windows",
    }
