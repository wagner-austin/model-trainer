from __future__ import annotations

import logging
import os
import sys
from typing import Final

import redis
import rq

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
        ok = bool(getattr(conn, "ping", lambda: True)())
    except Exception as e:  # pragma: no cover - connectivity check only
        logging.getLogger(__name__).warning("redis_ping_failed: %s", e)
        ok = False
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

    # RQ 1.16.x: allow global connection helpers when present
    push_conn = getattr(rq, "push_connection", None)
    pop_conn = getattr(rq, "pop_connection", None)
    if callable(push_conn):
        push_conn(conn)
    try:
        q = rq.Queue(queue_name, connection=conn)
        w = rq.Worker(queues=[q], connection=conn)
        success = bool(w.work())
        return 0 if success else 1
    finally:
        if callable(pop_conn):
            pop_conn()


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future flags; keep signature stable
    try:
        return run()
    except Exception as e:
        logging.getLogger(__name__).exception("worker_failed: %s", e)
        return 1


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main(sys.argv[1:]))
