from __future__ import annotations

from pathlib import Path

from model_trainer.core.logging.service import LoggingService


def test_attach_run_file_deduplicates_handlers(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    svc = LoggingService.create()

    # Attach twice, log twice
    adapter1 = svc.attach_run_file(
        path=str(log_path), category="test", service="logger", run_id="r1"
    )
    adapter2 = svc.attach_run_file(
        path=str(log_path), category="test", service="logger", run_id="r1"
    )
    adapter1.info("one", extra={"event": "e1"})
    adapter2.info("two", extra={"event": "e2"})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    # Expect exactly two lines, not four
    assert len(lines) == 2
