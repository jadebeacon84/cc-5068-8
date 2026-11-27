"""Rebuild the release package and verify quarantine/restore by interval id.

    python3 -m scripts.build_release [--data-root DIR] [--release-root DIR]

The working dataset is copied, never modified. On the copy, every overlapping
audited pair goes through a source-track quarantine/restore cycle; the final
package must pass all label checks. Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json

from ser.config import DATA_ROOT, DEFAULT_CONFIG, EXPERIMENTS_ROOT
from ser.release import build_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--release-root",
                        default=str(EXPERIMENTS_ROOT / "release"))
    args = parser.parse_args()

    summary = build_release(
        args.data_root, args.release_root, DEFAULT_CONFIG.audio)
    print(f"[build_release] release root: {summary['release_root']}")
    print(f"[build_release] recordings: {summary['recordings']}")
    print(f"[build_release] quarantine/restore cycles: {summary['n_cycles']}")
    for cycle in summary["quarantine_cycles"]:
        print(f"  {cycle['split']}/{cycle['recording_id']} "
              f"{cycle['event_id']}: {cycle['checks']}")
    print("[build_release] final label check:",
          "PASS" if summary["final_check"]["ok"] else "FAIL")
    print("[build_release] all identity cycles:",
          "PASS" if summary["all_cycles_ok"] else "FAIL")
    print("[build_release] report: "
          f"{args.release_root}/release_report.json")
    ok = summary["final_check"]["ok"] and summary["all_cycles_ok"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
