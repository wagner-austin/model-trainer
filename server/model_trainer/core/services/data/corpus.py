from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path


def list_text_files(root: str) -> list[str]:
    p = Path(root)
    if p.is_file():
        return [str(p)]
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".txt", ".text")):
                paths.append(str(Path(dirpath) / name))
    return paths


def iter_lines(files: Sequence[str]) -> Iterator[str]:
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s
