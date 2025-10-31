#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"


def run(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def main() -> int:
    # Ensure lock is current and environment matches lock (no fallbacks)
    print("[lint] poetry lock")
    if run(["poetry", "lock"], SERVER) != 0:
        return 1
    print("[lint] poetry sync --with dev")
    if run(["poetry", "sync", "--with", "dev"], SERVER) != 0:
        return 1
    # Ruff fix + format (via poetry env)
    print("[lint] ruff --fix")
    if run(["poetry", "run", "ruff", "check", ".", "--fix"], SERVER) != 0:
        return 1
    print("[lint] ruff format")
    if run(["poetry", "run", "ruff", "format", "."], SERVER) != 0:
        return 1
    # Mypy strict (always run; fail if deps missing)
    print("[lint] mypy --strict")
    if run(["poetry", "run", "mypy", "."], SERVER) != 0:
        return 1
    # Guards
    print("[lint] guards")
    if run([sys.executable, str(ROOT / "scripts" / "guard.py")], ROOT) != 0:
        return 1
    print("[lint] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
