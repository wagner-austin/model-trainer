#!/usr/bin/env python
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent

# Guard scope: scan the actual server codebase (and allow tests for most checks)
TARGET_DIRS = [
    ROOT / "server" / "model_trainer",
    ROOT / "server" / "tests",
]

EXCLUDE_DIRNAMES = {
    ".venv",
    "__pycache__",
    "node_modules",
    "typings",  # allow local stubs
}

ALLOW_EXT = {".py", ".pyi"}

# Patterns to flag as violations
PATTERNS: dict[str, re.Pattern[str]] = {
    "typing.Any": re.compile(r"\btyping\.Any\b"),
    "Any import": re.compile(r"\bfrom\s+typing\s+import\b[^#\n]*\bAny\b"),
    "Any usage": re.compile(r"(?<!\w)Any(?!\w)"),
    "type: ignore": re.compile(r"type:\s*ignore"),
    "typing.cast": re.compile(r"\btyping\.cast\b"),
    # Drift markers / ad-hoc code
    "TODO": re.compile(r"\bTODO\b"),
    "FIXME": re.compile(r"\bFIXME\b"),
    "HACK": re.compile(r"\bHACK\b"),
    "XXX": re.compile(r"\bXXX\b"),
    "WIP": re.compile(r"\bWIP\b"),
    # Print usage in library code
    "print()": re.compile(r"(^|\s)print\s*\("),
    # Frozen dataclasses are discouraged (harder to monkeypatch/tests)
    "dataclass(frozen=True)": re.compile(r"@dataclass\(\s*frozen\s*=\s*True\s*\)"),
    # Disallow ad-hoc global logging config; use central logging setup
    "logging.basicConfig": re.compile(r"\blogging\.basicConfig\s*\("),
    # Disallow noqa suppressions in code/tests to keep lints actionable
    "noqa": re.compile(r"#\s*noqa\b"),
}


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for base in paths:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in ALLOW_EXT:
                continue
            if any(part in EXCLUDE_DIRNAMES for part in p.parts):
                continue
            yield p


def scan_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # pragma: no cover
        return [f"{path}: failed to read: {e}"]

    lines = text.splitlines()
    for name, pat in PATTERNS.items():
        for i, line in enumerate(lines, start=1):
            if pat.search(line):
                errors.append(f"{path}:{i}: disallowed pattern: {name}")

    # Detect silent exception blocks: simple pass/ellipsis
    for i, line in enumerate(lines, start=1):
        if re.search(r"^\s*except(\b|\s).*:\s*$", line):
            # Look ahead to next non-empty line at greater indent
            j = i
            while j < len(lines):
                j += 1
                nxt = lines[j - 1] if j - 1 < len(lines) else ""
                if nxt.strip() == "":
                    continue
                # If the next significant line is pass or ellipsis, flag
                if re.match(r"^\s+(pass|\.\.\.)\s*(#.*)?$", nxt):
                    errors.append(f"{path}:{i}: disallowed pattern: silent except body")
                break
    # Detect broader silent exception blocks: no raise and no logging in block
    log_call_named = re.compile(r"\b(logging|log|logger)\.(debug|info|warning|error|exception|critical)\(")
    log_call_any = re.compile(r"\.(debug|info|warning|error|exception|critical)\(")
    raise_re = re.compile(r"\braise\b")
    for i0, line in enumerate(lines, start=1):
        m = re.match(r"^(\s*)except(\s+([^:]+))?:\s*$", line)
        if not m:
            continue
        except_indent = len(m.group(1))
        types = (m.group(3) or "").strip()
        broad = types == "" or "Exception" in types or "BaseException" in types
        # Find first line of body
        start = i0 + 1
        body_start = None
        while start <= len(lines):
            if start > len(lines):
                break
            sline = lines[start - 1]
            if sline.strip() == "":
                start += 1
                continue
            body_indent = len(re.match(r"^\s*", sline).group(0))
            body_start = start
            break
        if body_start is None:
            # Empty body already flagged above, skip
            continue
        # Collect body until dedent to except indent or next clause
        has_log = False
        has_raise = False
        j = body_start
        while j <= len(lines):
            cur = lines[j - 1]
            if cur.strip() == "":
                j += 1
                continue
            cur_indent = len(re.match(r"^\s*", cur).group(0))
            if cur_indent <= except_indent and re.match(r"^\s*(except\b|finally\b|else\b|$)", cur):
                break
            if raise_re.search(cur):
                has_raise = True
            if log_call_named.search(cur) or log_call_any.search(cur):
                has_log = True
            j += 1
        if broad:
            if not (has_log and has_raise):
                errors.append(f"{path}:{i0}: disallowed pattern: broad except requires log and raise")
        else:
            if not (has_log or has_raise):
                errors.append(f"{path}:{i0}: disallowed pattern: except block without log/raise")
    # Enforce Pydantic-first policy: no @dataclass in contracts/config
    if ("model_trainer" in path.parts and "core" in path.parts) and (
        "contracts" in path.parts or "config" in path.parts
    ):
        for i, line in enumerate(lines, start=1):
            if re.match(r"^\s*@dataclass\b", line):
                errors.append(f"{path}:{i}: disallowed pattern: dataclass in contracts/config")
            if re.search(r"from\s+dataclasses\s+import\s+dataclass\b", line):
                errors.append(f"{path}:{i}: disallowed pattern: dataclass import in contracts/config")
    return errors


def main() -> int:
    violations: list[str] = []
    for f in iter_files(TARGET_DIRS):
        violations.extend(scan_file(f))
    if violations:
        print("Guard checks failed:")
        for v in violations:
            print(f"  {v}")
        return 2
    print("Guards OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
