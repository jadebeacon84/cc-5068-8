"""Failure feedback loop: UI reports bad predictions into a local JSONL file.

Feedback stays on disk and is re-read by the inspection tooling. The web layer
never calls a remote recognition service; failures route back into the sound
event pipeline (cue sheet status + structured feedback log).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def append_feedback(path: Path, recording_id: str, uid: str,
                    reason: str, expected_class: str | None,
                    window_indices: list[int] | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "recording_id": recording_id,
        "event_uid": uid,
        "reason": reason,
        "expected_class": expected_class,
        "window_indices": window_indices or [],
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_feedback(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def feedback_summary(path: Path) -> dict:
    rows = read_feedback(path)
    by_reason: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        key = row.get("expected_class") or "unspecified"
        by_class[key] = by_class.get(key, 0) + 1
    return {"count": len(rows), "by_reason": by_reason,
            "by_expected_class": by_class}
