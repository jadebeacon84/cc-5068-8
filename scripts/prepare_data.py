"""Generate the annotated local dataset (run from repo root).

    python3 -m scripts.prepare_data
"""
from __future__ import annotations

import json

from ser.config import DATA_ROOT, DEFAULT_CONFIG
from ser.dataset import assert_no_leakage, read_manifest
from ser.synthesis import build_dataset


def main() -> None:
    cfg = DEFAULT_CONFIG
    manifest = build_dataset(DATA_ROOT, cfg.dataset)
    assert_no_leakage(manifest)
    cfg.save(DATA_ROOT / "config_snapshot.json")
    counts = {split: len(ids) for split, ids in manifest.items()}
    print("[prepare_data] wrote recordings:", json.dumps(counts))
    print("[prepare_data] leakage check: PASS (recording-level splits)")
    print(f"[prepare_data] data root: {DATA_ROOT}")
    stored = read_manifest(DATA_ROOT)
    total_events = 0
    for split, ids in stored.items():
        for rid in ids:
            labels = json.loads(
                (DATA_ROOT / split / rid / "labels.json").read_text()
            )
            total_events += len(labels["events"])
    print(f"[prepare_data] total annotated events: {total_events}")


if __name__ == "__main__":
    main()
