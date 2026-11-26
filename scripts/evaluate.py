"""Event-level evaluation on held-out and interference recordings.

    python3 -m scripts.evaluate

Adjudication is event-level (bipartite IoU). The script also prints the
inspectable feature/training/window configuration and re-runs the leakage
guard before scoring.
"""
from __future__ import annotations

import json

from ser.audio import audio_config_summary
from ser.config import EXPERIMENTS_ROOT
from ser.dataset import assert_no_leakage
from ser.pipeline import Pipeline


def main() -> None:
    pipe = Pipeline()
    manifest = pipe.require_data()
    assert_no_leakage(manifest)
    model = pipe.load_model()

    inspectable = {
        "audio": audio_config_summary(pipe.cfg.audio),
        "splits": {split: ids for split, ids in manifest.items()},
        "leakage_rule": "whole recording is the split unit; no slice crosses",
        "model_metadata": model.metadata,
    }
    out_dir = EXPERIMENTS_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inspectable_config.json").write_text(
        json.dumps(inspectable, indent=2, ensure_ascii=False)
    )

    for split in ("test", "interference"):
        summary = pipe.run_split_evaluation(split)
        print(f"\n=== {split} (event-level) ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[evaluate] reports written under {out_dir}")


if __name__ == "__main__":
    main()
