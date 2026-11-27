"""Label taxonomy maintenance: legacy mapping, listening audit, quarantine.

The project's target taxonomy used to collapse the mechanical bicycle bell and
the electronic prompt tone into one shared label, ``bell``. The current
taxonomy splits them into ``bell`` and ``chime``. This module owns everything
needed to keep old and new labels consistent:

  * a one-to-many legacy mapping (one legacy label may resolve to several
    current classes) plus the reverse lookup used by tooling;
  * per-interval audit state — audited intervals may enter single-class
    training packs, pending/unclaimed intervals may not;
  * legacy annotation spans that are either single events or whole segments,
    with coverage checks over the current intervals;
  * reversible source-track quarantine: moving the dry tracks away drops the
    affected intervals to ``unclaimed`` so they leave the training pack, and
    restoring them brings the audited identities back;
  * content digests so duplicate material and missing files are detectable.

Nothing in this module trains or infers; it only maintains label material.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import (
    AUDIT_AUDITED,
    AUDIT_PENDING,
    AUDIT_STATUSES,
    AUDIT_UNCLAIMED,
    LABEL_SCOPES,
    LEGACY_BELL_CLASS,
    SCOPE_SEGMENT,
    SCOPE_SINGLE,
    TARGET_CLASSES,
)

# One legacy label can map to one or many current classes (one-to-many).
LEGACY_TO_CURRENT: dict[str, tuple[str, ...]] = {
    "knock": ("knock",),
    "bell": ("bell", "chime"),
    "machine": ("machine",),
}
CURRENT_TO_LEGACY: dict[str, str] = {
    cls: legacy
    for legacy, current in LEGACY_TO_CURRENT.items()
    for cls in current
}

# Legacy material with an unresolved split stays explicitly uncertain; it is
# never silently assigned to one of the split classes.
UNRESOLVED_CLASS = "unresolved"


@dataclass(frozen=True)
class LegacyResolution:
    legacy_class: str
    candidates: tuple[str, ...]
    resolved: bool
    note: str = ""


def resolve_legacy_label(legacy_class: str,
                         current_class: str | None = None) -> LegacyResolution:
    """Map one legacy label onto its current candidates.

    When ``current_class`` is supplied and is a valid candidate the resolution
    is definite; otherwise the split class is reported as unresolved.
    """
    if legacy_class not in LEGACY_TO_CURRENT:
        return LegacyResolution(
            legacy_class, (), False,
            f"unknown legacy class {legacy_class!r}",
        )
    candidates = LEGACY_TO_CURRENT[legacy_class]
    if len(candidates) == 1:
        return LegacyResolution(legacy_class, candidates, True)
    if current_class in candidates:
        return LegacyResolution(
            legacy_class, candidates, True,
            f"listening review assigned {current_class}",
        )
    return LegacyResolution(
        legacy_class, candidates, False,
        f"legacy {legacy_class!r} split into {list(candidates)}; "
        "listening review required",
    )


def legacy_candidates_for(legacy_class: str) -> tuple[str, ...]:
    return LEGACY_TO_CURRENT.get(legacy_class, ())


def reverse_legacy_label(current_class: str) -> str:
    return CURRENT_TO_LEGACY.get(current_class, current_class)


# ------------------------------------------------------------- per-interval
def is_trainable_event(event: dict) -> bool:
    """Whether one labels.json event may feed a single-class training window."""
    if event.get("class") not in TARGET_CLASSES:
        return False
    if event.get("audit_status") != AUDIT_AUDITED:
        return False
    if event.get("identity_uncertain"):
        return False
    source = event.get("source")
    if not source or not Path(source).name:
        return False
    return True


def trainable_events(events: list[dict]) -> list[dict]:
    return [e for e in events if is_trainable_event(e)]


def events_by_audit(events: list[dict]) -> dict[str, list[dict]]:
    buckets = {status: [] for status in AUDIT_STATUSES}
    for event in events:
        status = event.get("audit_status", AUDIT_PENDING)
        buckets.setdefault(status, []).append(event)
    return buckets


def set_audit_status(events: list[dict], event_id: str,
                     status: str, note: str | None = None) -> dict:
    if status not in AUDIT_STATUSES:
        raise ValueError(f"unknown audit status {status!r}")
    for event in events:
        if event["event_id"] == event_id:
            event["audit_status"] = status
            if status == AUDIT_AUDITED:
                event["identity_uncertain"] = False
            if note is not None:
                event["audit_note"] = note
            return event
    raise KeyError(event_id)


# --------------------------------------------------------- legacy coverage
def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def check_legacy_coverage(events: list[dict],
                          legacy_entries: list[dict],
                          tolerance: float = 0.05) -> dict:
    """Verify every legacy span maps to current intervals correctly.

    Single-event legacy spans must map 1:1 to one current event of a candidate
    class with matching boundaries. Whole-segment legacy spans must cover one
    or more current events whose classes are all candidates of the legacy
    label; split-class segments additionally record whether the listening
    review resolved every covered event.
    """
    report = {"single_event": [], "whole_segment": [], "errors": []}
    for entry in legacy_entries:
        scope = entry.get("scope", SCOPE_SINGLE)
        if scope not in LABEL_SCOPES:
            report["errors"].append(
                f"legacy entry {entry.get('legacy_id')} has bad scope {scope!r}")
            continue
        candidates = legacy_candidates_for(entry["legacy_class"])
        covered = [
            e for e in events
            if _overlap(e["start"], e["end"], entry["start"], entry["end"]) > 0
        ]
        if not covered:
            report["errors"].append(
                f"legacy {entry.get('legacy_id')} covers no current event")
            continue
        bad_class = [e["event_id"] for e in covered
                     if e.get("class") not in candidates]
        if bad_class:
            report["errors"].append(
                f"legacy {entry.get('legacy_id')} covers non-candidate "
                f"events {bad_class}")
        if scope == SCOPE_SINGLE:
            exact = [
                e for e in events
                if e.get("class") in candidates
                and abs(e["start"] - entry["start"]) <= tolerance
                and abs(e["end"] - entry["end"]) <= tolerance
            ]
            ok = len(exact) == 1 and not bad_class
            report["single_event"].append({
                "legacy_id": entry.get("legacy_id"),
                "event_id": exact[0]["event_id"] if exact else None,
                "ok": bool(ok),
            })
            if not ok:
                report["errors"].append(
                    f"legacy single-event {entry.get('legacy_id')} does not "
                    "map 1:1 with matching boundaries")
        else:
            unresolved = [
                e["event_id"] for e in covered
                if e.get("identity_uncertain")
                or e.get("audit_status") != AUDIT_AUDITED
            ]
            report["whole_segment"].append({
                "legacy_id": entry.get("legacy_id"),
                "event_ids": [e["event_id"] for e in covered],
                "classes": sorted({e["class"] for e in covered}),
                "resolved": not unresolved,
                "unresolved_event_ids": unresolved,
            })
            if bad_class:
                continue
    return report


# --------------------------------------------------------------- digests
def sha256_file(path: Path | str, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def recording_digests(rec_dir: Path | str) -> dict[str, str]:
    """SHA-256 of mix.wav and every dry source track under the recording."""
    rec_dir = Path(rec_dir)
    out = {}
    mix = rec_dir / "mix.wav"
    if mix.exists():
        out["mix.wav"] = sha256_file(mix)
    for wav in sorted((rec_dir / "sources").glob("*.wav")):
        out[f"sources/{wav.name}"] = sha256_file(wav)
    return out


def find_duplicate_material(per_recording: dict[str, dict[str, str]]) -> list[str]:
    """Source tracks reused across different recordings are duplicate material.

    Match on content digest alone: duplicated audio is a leak even when the
    copied track was given a different event-id file name.
    """
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[str] = []
    for rid, digests in per_recording.items():
        for rel_path, digest in digests.items():
            if not rel_path.startswith("sources/"):
                continue
            if digest in seen and seen[digest][0] != rid:
                prev_rid, prev_path = seen[digest]
                duplicates.append(
                    f"identical source audio {rel_path} in {rid} and "
                    f"{prev_path} in {prev_rid}")
            else:
                seen[digest] = (rid, rel_path)
    return duplicates


# ------------------------------------------------- source-track quarantine
def quarantine_source_tracks(rec_dir: Path | str,
                             event_ids: list[str] | None = None,
                             quarantine_dirname: str = "sources_quarantined"
                             ) -> dict:
    """Move dry source tracks out of ``sources/`` and downgrade their events.

    Intervals whose tracks can no longer be opened drop to ``unclaimed`` and
    therefore leave every single-class training pack. The move is reversible
    via :func:`restore_source_tracks`. Returns the affected event records.
    """
    rec_dir = Path(rec_dir)
    labels_path = rec_dir / "labels.json"
    labels = json.loads(labels_path.read_text())
    sources_dir = rec_dir / "sources"
    quarantine_dir = rec_dir / quarantine_dirname
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    targets = {e["event_id"]: e for e in labels["events"]
               if event_ids is None or e["event_id"] in event_ids}
    moved: list[dict] = []
    for event in labels["events"]:
        if event["event_id"] not in targets:
            continue
        source_rel = event.get("source")
        if not source_rel:
            continue
        src = rec_dir / source_rel
        if not src.exists():
            continue
        dest = quarantine_dir / Path(source_rel).name
        src.rename(dest)
        previous = event.get("audit_status")
        event["audit_status"] = AUDIT_UNCLAIMED
        event["identity_uncertain"] = True
        event["quarantined"] = True
        event["previous_audit_status"] = previous
        moved.append({
            "event_id": event["event_id"],
            "class": event["class"],
            "source": source_rel,
            "previous_audit_status": previous,
        })
    labels["quarantine"] = {
        "dir": quarantine_dirname,
        "moved": moved,
    }
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    return {"labels": labels, "moved": moved}


def restore_source_tracks(rec_dir: Path | str,
                          quarantine_dirname: str = "sources_quarantined"
                          ) -> dict:
    """Undo a quarantine: move tracks back and restore the prior audit state."""
    rec_dir = Path(rec_dir)
    labels_path = rec_dir / "labels.json"
    labels = json.loads(labels_path.read_text())
    quarantine = labels.get("quarantine")
    if not quarantine:
        return {"labels": labels, "restored": []}
    quarantine_dir = rec_dir / quarantine.get("dir", quarantine_dirname)
    sources_dir = rec_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    restored: list[dict] = []
    moved_ids = {m["event_id"]: m for m in quarantine.get("moved", [])}
    for event in labels["events"]:
        record = moved_ids.get(event["event_id"])
        if record is None:
            continue
        quarantined = quarantine_dir / Path(record["source"]).name
        if not quarantined.exists():
            continue
        target = rec_dir / record["source"]
        target.parent.mkdir(parents=True, exist_ok=True)
        quarantined.rename(target)
        event["audit_status"] = record.get(
            "previous_audit_status", AUDIT_AUDITED)
        event["identity_uncertain"] = event["audit_status"] != AUDIT_AUDITED
        event.pop("quarantined", None)
        event.pop("previous_audit_status", None)
        restored.append({
            "event_id": event["event_id"],
            "class": event["class"],
            "source": record["source"],
            "audit_status": event["audit_status"],
        })
    labels.pop("quarantine", None)
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    if quarantine_dir.exists() and not any(quarantine_dir.iterdir()):
        quarantine_dir.rmdir()
    return {"labels": labels, "restored": restored}


def missing_source_events(rec_dir: Path | str, events: list[dict]) -> list[str]:
    """Event ids whose declared dry source track is not on disk."""
    rec_dir = Path(rec_dir)
    return [
        e["event_id"] for e in events
        if e.get("source") and not (rec_dir / e["source"]).exists()
    ]


def build_legacy_seed(legacy_class: str, start: float, end: float,
                      scope: str, legacy_id: str,
                      note: str = "") -> dict:
    """Helper for authoring a legacy annotation span in labels.json."""
    if scope not in LABEL_SCOPES:
        raise ValueError(f"bad legacy scope {scope!r}")
    return {
        "legacy_id": legacy_id,
        "legacy_class": legacy_class,
        "start": round(start, 3),
        "end": round(end, 3),
        "scope": scope,
        "candidate_classes": list(legacy_candidates_for(legacy_class)),
        "note": note,
    }


__all__ = [
    "LEGACY_TO_CURRENT",
    "CURRENT_TO_LEGACY",
    "UNRESOLVED_CLASS",
    "LegacyResolution",
    "resolve_legacy_label",
    "legacy_candidates_for",
    "reverse_legacy_label",
    "is_trainable_event",
    "trainable_events",
    "events_by_audit",
    "set_audit_status",
    "check_legacy_coverage",
    "sha256_file",
    "recording_digests",
    "find_duplicate_material",
    "quarantine_source_tracks",
    "restore_source_tracks",
    "missing_source_events",
    "build_legacy_seed",
    "SCOPE_SINGLE",
    "SCOPE_SEGMENT",
    "LEGACY_BELL_CLASS",
]
