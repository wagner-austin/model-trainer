from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from logging import Handler, Logger, LoggerAdapter
from pathlib import Path

from .types import LoggingExtra


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

        # Dynamically include all extra fields defined in LoggingExtra TypedDict
        # Special handling for category and service since they might be in static_fields
        for field_name in LoggingExtra.__annotations__:
            if hasattr(record, field_name) and field_name not in self._static:
                value: object = getattr(record, field_name)
                payload[field_name] = value

        return json.dumps(payload, separators=(",", ":"))


_HANDLERS: dict[str, Handler] = {}


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
        abs_path = str(Path(path).resolve())
        handler = _HANDLERS.get(abs_path)
        if handler is None:
            handler = logging.FileHandler(abs_path, encoding="utf-8")
            handler.setFormatter(
                _JsonFormatter(static_fields={"category": category, "service": service})
            )
            self.base_logger.addHandler(handler)
            _HANDLERS[abs_path] = handler
        return self.adapter(
            category=category, service=service, run_id=run_id, tokenizer_id=tokenizer_id
        )

    def close_run_file(self: LoggingService, *, path: str) -> None:
        abs_path = str(Path(path).resolve())
        handler = _HANDLERS.get(abs_path)
        if handler is not None:
            try:
                self.base_logger.removeHandler(handler)
            finally:
                try:
                    handler.close()
                finally:
                    _HANDLERS.pop(abs_path, None)
