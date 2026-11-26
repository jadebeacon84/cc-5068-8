"""Train the local multi-label model and store a reproducible artifact.

    python3 -m scripts.train_model
"""
from __future__ import annotations

import json

from ser.config import EXPERIMENTS_ROOT
from ser.pipeline import Pipeline


def main() -> None:
    pipe = Pipeline()
    metadata = pipe.train()
    out = EXPERIMENTS_ROOT / "model"
    print(f"[train_model] artifact saved to {out}")
    print("[train_model] calibrated thresholds:",
          json.dumps(metadata["thresholds"], ensure_ascii=False))
    print(f"[train_model] train windows: {metadata['train_windows']}")
    print(f"[train_model] train recordings: {len(metadata['train_recordings'])}"
          f", val recordings: {len(metadata['val_recordings'])}")


if __name__ == "__main__":
    main()
