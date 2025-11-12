from __future__ import annotations

from pathlib import Path

from model_trainer.core.logging.service import LoggingService


def test_close_run_file_no_handler_is_noop(tmp_path: Path) -> None:
    # Path that has never been attached
    log_path = tmp_path / "never.jsonl"
    svc = LoggingService.create()
    # Should not raise and should not create the file
    svc.close_run_file(path=str(log_path))
    assert not log_path.exists()
