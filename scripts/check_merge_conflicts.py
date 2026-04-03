#!/usr/bin/env python3
"""Fail if git conflict markers are present in tracked text files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def tracked_files() -> list[Path]:
    proc = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    offenders: list[tuple[Path, int, str]] = []
    for file in tracked_files():
        try:
            content = file.read_text(encoding="utf-8")
        except Exception:
            continue

        for idx, line in enumerate(content.splitlines(), start=1):
            if line.startswith(MARKERS):
                offenders.append((file, idx, line))

    if offenders:
        print("Merge conflict markers detected:")
        for path, line_no, text in offenders:
            print(f"- {path}:{line_no}: {text}")
        return 1

    print("No merge conflict markers found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
