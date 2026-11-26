"""Inspect splits, analysis windows and feature shapes; enforce no leakage.

    python3 -m scripts.check_splits
"""
from __future__ import annotations

from collections import Counter

from ser.config import TARGET_CLASSES
from ser.dataset import (
    assert_no_leakage,
    load_split,
    recording_patches,
)
from ser.pipeline import Pipeline


def main() -> None:
    pipe = Pipeline()
    manifest = pipe.require_data()
    assert_no_leakage(manifest)
    print("[check_splits] leakage guard: PASS")
    print("[check_splits] split sizes:",
          {k: len(v) for k, v in manifest.items()})

    fb = None
    for split in ("train", "val", "test", "interference"):
        recs = load_split(pipe.data_root, split, pipe.cfg.audio, fb)
        if fb is None:
            fb = recs[0].filterbank
        label_counter: Counter = Counter()
        shapes = set()
        window_count = 0
        for rec in recs:
            patches = recording_patches(rec, pipe.cfg.audio.win_frames)
            shapes.add(patches.shape)
            window_count += len(rec.window_times)
            for row in rec.window_labels:
                if row.sum() == 0:
                    label_counter["background_or_unknown"] += 1
                for c, flag in enumerate(row):
                    if flag:
                        label_counter[TARGET_CLASSES[c]] += 1
        print(f"\n[{split}] recordings={len(recs)} windows={window_count}")
        print(f"  window patch shapes: {sorted(shapes)}")
        print(f"  window labels: {dict(label_counter)}")
        print(f"  recording ids: {[r.recording_id for r in recs]}")


if __name__ == "__main__":
    main()
