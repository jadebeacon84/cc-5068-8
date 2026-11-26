"""Demonstrate the cut -> cue merge -> restore -> re-split cycle.

    python3 -m scripts.cut_restore_demo

The script scans recordings for two same-class markers with a clean gap (the
overlapping/ringing bell case), simulates the cut on a copy until it finds a
recording where the two markers genuinely merge, then performs the real
reversible cut, restores the section and verifies the original markers against
the dry source tracks.
"""
from __future__ import annotations

from copy import deepcopy

from ser.config import TARGET_CLASSES
from ser.cue import _merge_touching
from ser.pipeline import Pipeline


def _candidate_pairs(entries: list[dict]):
    pairs = []
    for cls in TARGET_CLASSES:
        rows = sorted((e for e in entries if e["cls"] == cls),
                      key=lambda e: e["start"])
        for a, b in zip(rows, rows[1:]):
            gap = b["start"] - a["end"]
            if not (0.05 < gap < 1.5):
                continue
            blocked = any(
                e is not a and e is not b
                and e["start"] >= a["end"] and e["end"] <= b["start"]
                for e in entries
            )
            if not blocked:
                pairs.append((a, b, gap))
    return pairs


def _simulate_cut(entries: list[dict], start: float, end: float) -> list[dict]:
    """Mirror of CueSheet.cut_section geometry without touching persisted state."""
    shift = end - start
    rows = deepcopy(entries)
    for row in rows:
        a, b = row["start"], row["end"]
        if b <= start:
            continue
        if a >= end:
            row["start"] = round(a - shift, 3)
            row["end"] = round(b - shift, 3)
        elif a >= start and b <= end:
            row["start"] = row["end"] = round(start, 3)
        elif a < start and b <= end:
            row["end"] = round(start, 3)
        elif a >= start and b > end:
            row["start"] = round(start, 3)
            row["end"] = round(start + max(0.0, b - end), 3)
        else:
            row["end"] = round(b - shift, 3)
    return _merge_touching(rows, gap_tol=0.05)


def main() -> None:
    pipe = Pipeline()
    pipe.load_model()
    pipe.load_recordings()
    pipe.cue_sheet.entries.clear()
    pipe.cue_sheet.cuts.clear()

    chosen = None
    for rec_info in pipe.list_recordings():
        rid = rec_info["recording_id"]
        pipe.run_inference(rid, force=True)
        entries = pipe.cue_sheet.entries[rid]
        for a, b, gap in _candidate_pairs(entries):
            simulated = _simulate_cut(entries, a["end"], b["start"])
            merged = any(e.get("cue_status") == "merged_by_cut"
                         for e in simulated)
            if merged:
                chosen = (rid, a, b, gap)
                break
        if chosen:
            break
    if chosen is None:
        raise SystemExit("no adjacent same-class pair produces a merge")

    rid, a, b, gap = chosen
    print(f"[demo] recording: {rid}")
    print(f"[demo] pair: {a['event_uid']} end={a['end']:.3f} -> "
          f"{b['event_uid']} start={b['start']:.3f} (gap={gap:.3f}s)")

    cut_start, cut_end = round(a["end"], 3), round(b["start"], 3)
    before = len(pipe.cue_sheet.entries[rid])
    result = pipe.cue_sheet.cut_section(rid, cut_start, cut_end)
    after_cut = len(result["entries"])
    print(f"[demo] cut [{cut_start}, {cut_end}]: markers {before} -> {after_cut}")
    assert after_cut < before, "cut was expected to merge markers"
    assert any(e.get("cue_status") == "merged_by_cut"
               for e in result["entries"])

    restored = pipe.cue_sheet.restore_section(rid)
    after_restore = len(restored["entries"])
    print(f"[demo] restore: markers -> {after_restore}")
    assert after_restore == before, "restore must return original markers"

    rec = pipe.get_recording(rid)
    sources = [e for e in rec.labels["events"] if e["class"] in TARGET_CLASSES]
    report = pipe.cue_sheet.verify_with_sources(rid, sources)
    print("[demo] source verification:")
    for m in report["matched"]:
        print(f"  {m['uid']} -> {m['source_event_id']} "
              f"iou={m['iou']} class_match={m['class_match']} "
              f"boundary={m['boundary_error']}")
    print(f"[demo] all identities exact: {report['all_identities_exact']}")
    print(f"[demo] cue sheet: {pipe.cue_sheet.path}")


if __name__ == "__main__":
    main()
