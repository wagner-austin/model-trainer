from __future__ import annotations

import logging

from model_trainer.core.logging.service import LoggingService, _JsonFormatter


def test_logging_adapter_fields_and_formatter() -> None:
    svc = LoggingService.create()
    _ = svc.adapter(category="api", service="runs", run_id="r1", tokenizer_id="t1")

    # Build a record and format with static fields present
    rec = logging.LogRecord(
        name="model_trainer",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    # Inject expected extra attributes
    rec.category = "api"
    rec.service = "runs"
    rec.run_id = "r1"
    rec.tokenizer_id = "t1"
    rec.event = "e"
    rec.error_code = "X"

    fmt = _JsonFormatter(static_fields={"category": "api", "service": "runs"})
    out = fmt.format(rec)
    # Ensure fields are present in output JSON string
    for key in ("category", "service", "run_id", "tokenizer_id", "event", "error_code", "msg"):
        assert key in out
