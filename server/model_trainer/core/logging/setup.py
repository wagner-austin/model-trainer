from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self: JsonFormatter, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "instance": os.getenv("APP_INSTANCE_ID", _compute_instance_id()),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def _compute_instance_id() -> str:
    host = socket.gethostname().split(".")[0]
    pid = os.getpid()
    return f"{host}-{pid}"


def setup_logging(level: str = "INFO") -> None:
    levels: dict[str, int] = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    lvl = levels.get(level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers by default
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
