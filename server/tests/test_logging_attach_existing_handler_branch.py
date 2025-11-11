from __future__ import annotations

from pathlib import Path

from model_trainer.core.logging.service import LoggingService


def test_attach_run_file_reuses_existing_handler(tmp_path: Path) -> None:
    svc = LoggingService.create()
    path = str(tmp_path / "logs.jsonl")
    _ = svc.attach_run_file(path=path, category="c", service="s")
    _ = svc.attach_run_file(path=path, category="c", service="s")
