from __future__ import annotations

from pathlib import Path

from model_trainer.core.logging.service import LoggingService


def test_close_then_reattach_produces_expected_lines(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    svc = LoggingService.create()
    a1 = svc.attach_run_file(path=str(p), category="c", service="s", run_id="r")
    a1.info("one", extra={"event": "e1"})
    svc.close_run_file(path=str(p))

    a2 = svc.attach_run_file(path=str(p), category="c", service="s", run_id="r")
    a2.info("two", extra={"event": "e2"})

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
