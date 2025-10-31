from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from logging import Logger, LoggerAdapter
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def __init__(self: _JsonFormatter, *, static_fields: dict[str, str] | None = None) -> None:
        super().__init__()
        self._static = static_fields or {}

    def format(self: _JsonFormatter, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in self._static.items():
            payload[k] = v
        if hasattr(record, "category"):
            cat: object = record.category
            payload["category"] = cat
        if hasattr(record, "service"):
            svc: object = record.service
            payload["service"] = svc
        if hasattr(record, "run_id"):
            rid: object = record.run_id
            payload["run_id"] = rid
        if hasattr(record, "tokenizer_id"):
            tid: object = record.tokenizer_id
            payload["tokenizer_id"] = tid
        if hasattr(record, "event"):
            evt: object = record.event
            payload["event"] = evt
        if hasattr(record, "error_code"):
            ec: object = record.error_code
            payload["error_code"] = ec
        return json.dumps(payload, separators=(",", ":"))


@dataclass
class LoggingService:
    base_logger: Logger

    @classmethod
    def create(cls: type[LoggingService]) -> LoggingService:
        return cls(base_logger=logging.getLogger("model_trainer"))

    def adapter(
        self: LoggingService,
        *,
        category: str,
        service: str,
        run_id: str | None = None,
        tokenizer_id: str | None = None,
    ) -> LoggerAdapter[Logger]:
        extra: dict[str, str] = {"category": category, "service": service}
        if run_id is not None:
            extra["run_id"] = run_id
        if tokenizer_id is not None:
            extra["tokenizer_id"] = tokenizer_id
        return LoggerAdapter(self.base_logger, extra)

    def attach_run_file(
        self: LoggingService,
        *,
        path: str,
        category: str,
        service: str,
        run_id: str | None = None,
        tokenizer_id: str | None = None,
    ) -> LoggerAdapter[Logger]:
        os.makedirs(Path(path).parent, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            _JsonFormatter(static_fields={"category": category, "service": service})
        )
        self.base_logger.addHandler(handler)
        return self.adapter(
            category=category, service=service, run_id=run_id, tokenizer_id=tokenizer_id
        )
