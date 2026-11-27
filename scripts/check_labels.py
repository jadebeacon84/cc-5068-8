"""Validate label material: durations, splits, digests, legacy mapping, packing.

    python3 -m scripts.check_labels [--data-root DIR] [--json]

Runs entirely on disk artifacts and never edits data, so it can be pointed at
a freshly generated clean directory. Exits non-zero when any check fails.
"""
from __future__ import annotations

import argparse
import json

from ser.config import DATA_ROOT, DEFAULT_CONFIG
from ser.label_check import check_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT),
                        help="dataset root containing manifest.json")
    parser.add_argument("--json", action="store_true",
                        help="emit the full machine-readable report")
    args = parser.parse_args()

    report = check_dataset(args.data_root, DEFAULT_CONFIG.audio)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"[check_labels] root: {report.root}")
        print(f"[check_labels] recordings={report.stats.get('recordings')} "
              f"events={report.stats.get('events')} "
              f"audited={report.stats.get('audited')} "
              f"pending={report.stats.get('pending_review')} "
              f"unclaimed={report.stats.get('unclaimed')} "
              f"trainable={report.stats.get('trainable')}")
        print(f"[check_labels] short events: {report.stats.get('short_events')}")
        print(f"[check_labels] legacy entries: "
              f"{report.stats.get('legacy_entries')}")
        for message in report.warnings:
            print(f"  WARNING: {message}")
        for message in report.errors:
            print(f"  ERROR: {message}")
        print(f"[check_labels] result: {'PASS' if report.ok else 'FAIL'} "
              f"({len(report.errors)} errors, {len(report.warnings)} warnings)")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
