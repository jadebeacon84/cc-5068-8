"""Release-package builder with quarantine/restore verification by interval id.

The builder never mutates the working dataset: it copies the on-disk dataset
into a fresh release directory, runs the full label check there, then for each
overlapping same-legacy-label pair it exercises the reversible source-track
quarantine on the *copy*:

  * moving the dry track away must downgrade that exact interval to
    ``unclaimed`` and remove it from the single-class training pack;
  * restoring the track must bring back the original interval id, class,
    boundaries and audit state, and re-admit it to the pack.

A final clean check and a checksum manifest are written so the released
package is independently re-verifiable.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import DEFAULT_CONFIG
from .dataset import label_windows
from .label_check import check_dataset
from .labeling import (
    is_trainable_event,
    quarantine_source_tracks,
    restore_source_tracks,
)


def _overlap_pairs(events: list[dict]) -> list[tuple[dict, dict]]:
    target = [e for e in events if is_trainable_event(e)]
    pairs = []
    for i, a in enumerate(target):
        for b in target[i + 1:]:
            if min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0:
                pairs.append((a, b))
    return pairs


def _packed_ids(events: list[dict], audio_cfg) -> set[str]:
    grid = [t for t in _window_grid(events, audio_cfg)]
    rows = label_windows(events, grid, audio_cfg, audited_only=True)
    packed = set()
    for i, w_start in enumerate(grid):
        if rows[i].any():
            for ev in events:
                if (is_trainable_event(ev)
                        and min(w_start + audio_cfg.window_seconds, ev["end"])
                        - max(w_start, ev["start"])
                        >= audio_cfg.min_label_overlap):
                    packed.add(ev["event_id"])
    return packed


def _window_grid(events: list[dict], audio_cfg) -> list[float]:
    end = max((e["end"] for e in events), default=audio_cfg.window_seconds)
    n = int(end / audio_cfg.window_hop_seconds) + 2
    return [i * audio_cfg.window_hop_seconds for i in range(n)]


def verify_quarantine_cycle(rec_dir: Path, event_id: str,
                            audio_cfg, quarantine_dirname: str) -> dict:
    """Quarantine and restore one interval; assert identity by event id."""
    labels_path = rec_dir / "labels.json"
    before = json.loads(labels_path.read_text())
    original = next(e for e in before["events"] if e["event_id"] == event_id)
    if not is_trainable_event(original):
        raise AssertionError(f"{event_id} must be audited before quarantine")
    before_packed = _packed_ids(before["events"], audio_cfg)
    if event_id not in before_packed:
        raise AssertionError(f"{event_id} missing from training pack before cut")

    moved = quarantine_source_tracks(rec_dir, [event_id], quarantine_dirname)
    quarantined_labels = json.loads(labels_path.read_text())
    quarantined = next(e for e in quarantined_labels["events"]
                       if e["event_id"] == event_id)
    if quarantined["audit_status"] != "unclaimed":
        raise AssertionError(f"{event_id} was not downgraded to unclaimed")
    after_packed = _packed_ids(quarantined_labels["events"], audio_cfg)
    if event_id in after_packed:
        raise AssertionError(f"{event_id} still packed after source removal")
    if not (rec_dir / quarantine_dirname / Path(original["source"]).name).exists():
        raise AssertionError(f"{event_id} track not found in quarantine dir")

    restored = restore_source_tracks(rec_dir, quarantine_dirname)
    final_labels = json.loads(labels_path.read_text())
    final = next(e for e in final_labels["events"]
                 if e["event_id"] == event_id)
    checks = {
        "event_id_preserved": final["event_id"] == original["event_id"],
        "class_preserved": final["class"] == original["class"],
        "boundaries_preserved": (
            abs(final["start"] - original["start"]) < 1e-9
            and abs(final["end"] - original["end"]) < 1e-9
        ),
        "audit_restored": final["audit_status"] == original["audit_status"],
        "source_restored": (rec_dir / original["source"]).exists(),
        "repacked": event_id in _packed_ids(final_labels["events"], audio_cfg),
        "restored_record": next(
            (r for r in restored["restored"] if r["event_id"] == event_id), {}),
    }
    checks["ok"] = all(
        v for k, v in checks.items() if k not in ("restored_record",))
    return {"event_id": event_id, "moved": moved["moved"], "checks": checks}


def build_release(data_root: Path | str, release_root: Path | str,
                  audio_cfg=None) -> dict:
    data_root = Path(data_root)
    release_root = Path(release_root)
    audio_cfg = audio_cfg or DEFAULT_CONFIG.audio
    if release_root.exists():
        shutil.rmtree(release_root)
    release_root.parent.mkdir(parents=True, exist_ok=True)

    source_report = check_dataset(data_root, audio_cfg)
    if not source_report.ok:
        raise AssertionError(
            "source dataset fails label checks: "
            + "; ".join(source_report.errors))

    shutil.copytree(data_root, release_root,
                    ignore=shutil.ignore_patterns(
                        "cue_sheet.json", "feedback.jsonl",
                        "sources_quarantined"))
    manifest = json.loads((release_root / "manifest.json").read_text())

    cycles = []
    for split, ids in manifest.items():
        for rid in ids:
            rec_dir = release_root / split / rid
            labels = json.loads((rec_dir / "labels.json").read_text())
            for a, b in _overlap_pairs(labels["events"]):
                for event in (a, b):
                    cycles.append({
                        "split": split,
                        "recording_id": rid,
                        **verify_quarantine_cycle(
                            rec_dir, event["event_id"], audio_cfg,
                            DEFAULT_CONFIG.dataset.quarantine_dirname,
                        ),
                    })

    final_report = check_dataset(release_root, audio_cfg)
    if not final_report.ok:
        raise AssertionError(
            "release dataset fails final label checks: "
            + "; ".join(final_report.errors))
    if any(not c["checks"]["ok"] for c in cycles):
        raise AssertionError("quarantine/restore identity check failed")

    checksum_manifest = {}
    for wav in sorted(release_root.rglob("*.wav")):
        from .labeling import sha256_file
        checksum_manifest[str(wav.relative_to(release_root))] = sha256_file(wav)

    summary = {
        "data_root": str(data_root),
        "release_root": str(release_root),
        "recordings": sum(len(ids) for ids in manifest.values()),
        "quarantine_cycles": cycles,
        "n_cycles": len(cycles),
        "all_cycles_ok": all(c["checks"]["ok"] for c in cycles),
        "final_check": final_report.to_dict(),
        "checksums": checksum_manifest,
    }
    (release_root / "release_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
