from __future__ import annotations

import logging
import os
import os as _os
import shutil as _shutil
import sys
from typing import Final

import redis

from ..core.config.settings import Settings
from ..core.logging.setup import setup_logging


def _env_str(name: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"{name} is required")
    return v


def _redis_from_env(url: str) -> redis.Redis[bytes]:
    # RQ stores binary payloads (pickled); decode_responses must be False
    return redis.from_url(url, decode_responses=False)


def _assert_redis_ok(conn: redis.Redis[bytes]) -> None:
    try:
        ok = bool(conn.ping())
    except (redis.exceptions.RedisError, OSError, ValueError) as e:  # pragma: no cover
        logging.getLogger(__name__).warning("redis_ping_failed: %s", e)
        raise RuntimeError("Failed to connect to Redis") from e
    if not ok:
        raise RuntimeError("Failed to connect to Redis")


def _queue_name_from_settings(cfg: Settings) -> str:
    name = cfg.rq.queue_name.strip()
    if name == "":
        raise RuntimeError("RQ queue name is required (settings.rq.queue_name)")
    return name


def run() -> int:
    cfg = Settings()
    setup_logging(cfg.logging.level)

    # Prefer explicit REDIS_URL so this entrypoint works outside the API app
    url_env: Final[str] = os.getenv("REDIS_URL", cfg.redis.url)
    if url_env.strip() == "":
        raise RuntimeError("REDIS_URL is required")

    queue_name = _queue_name_from_settings(cfg)

    conn = _redis_from_env(url_env)
    _assert_redis_ok(conn)

    # Handoff to official RQ CLI to keep our code strictly typed without importing rq runtime APIs
    rq_bin = _shutil.which("rq")
    if rq_bin is None:
        raise RuntimeError("rq CLI not found in PATH; ensure RQ is installed in this environment")
    _os.execvp(rq_bin, ["rq", "worker", queue_name, "--with-scheduler"])  # never returns
    return 0


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future flags; keep signature stable
    try:
        return run()
    except KeyboardInterrupt:  # graceful shutdown signal
        return 130


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main(sys.argv[1:]))
