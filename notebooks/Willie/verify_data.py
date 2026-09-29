#!/usr/bin/env python3
"""
Verify that the wound datasets have been placed where WILLIE's manifests expect.

The images are not redistributed with this repository. Download FUSeg, AZH and
Medetec from their original sources (see README), place them under data/, then
run this from the repository root:

    python verify_data.py

It reports, per dataset directory, how many referenced files are present and
lists the first few that are missing. Read-only — changes nothing.
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

INDEX = Path("data/processed/index/data_index.csv")
PATH_COLUMNS = ("image_path", "mask_path")


def main():
    root = Path(".").resolve()
    if not INDEX.exists():
        print(f"ERROR: {INDEX} not found. Run this from the repository root.",
              file=sys.stderr)
        return 2

    present = defaultdict(int)
    missing = defaultdict(list)

    with INDEX.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            for col in PATH_COLUMNS:
                rel = (row.get(col) or "").strip()
                if not rel:
                    continue
                folder = os.path.dirname(rel)
                if (root / rel).exists():
                    present[folder] += 1
                else:
                    missing[folder].append(rel)

    folders = sorted(set(present) | set(missing))
    if not folders:
        print("No paths found in the index. Is the file intact?")
        return 2

    width = max(len(f) for f in folders)
    print(f"{'directory':{width}s} {'found':>7} {'missing':>8}")
    print("-" * (width + 17))

    total_ok = total_missing = 0
    for folder in folders:
        ok = present[folder]
        gone = len(missing[folder])
        total_ok += ok
        total_missing += gone
        flag = "" if gone == 0 else "   <-- incomplete"
        print(f"{folder:{width}s} {ok:7d} {gone:8d}{flag}")

    print("-" * (width + 17))
    print(f"{'TOTAL':{width}s} {total_ok:7d} {total_missing:8d}")

    if total_missing:
        print(f"\n{total_missing} referenced file(s) are missing. Examples:")
        shown = 0
        for folder in folders:
            for rel in missing[folder][:3]:
                print(f"  {rel}")
                shown += 1
                if shown >= 12:
                    break
            if shown >= 12:
                break
        print("\nCheck the expected layout in the README. Common causes:")
        print("  - dataset extracted one level too deep (an extra nested folder)")
        print("  - AZH class folder renamed; it must be 'no wound', with a space")
        print("  - FUSeg masks in 'masks/' rather than 'labels/'")
        return 1

    print("\nAll referenced files present. The splits will reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
