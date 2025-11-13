from __future__ import annotations

from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.api.routes.runs import _RunsRoutes
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_logs_stream_follow_none_max_loops_exercises_else_branch(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Real container instance with fakeredis to satisfy types without external deps
    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fakeredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr("redis.from_url", _fake_from_url)
    s = Settings()
    container: ServiceContainer = ServiceContainer.from_settings(s)

    # Prepare a log file with a single line
    log_path = tmp_path / "logs.jsonl"
    log_path.write_text("one\n", encoding="utf-8")

    # Create the routes helper with real container for logging
    h = _RunsRoutes(container)

    # Provide a sleep function that appends a line to the file to wake the follower
    def _sleep_and_append(_: float) -> None:
        log_path.write_text("two\n", encoding="utf-8")

    h._sleep_fn = _sleep_and_append
    h._follow_max_loops = None  # ensure the None branch at line 110 is evaluated

    # Consume SSE iterator: first yield is initial tail; second after sleep/appended line
    it = h._sse_iter(str(log_path), tail=1, follow=True)
    first = next(it)  # initial tail ("one")
    assert first.startswith(b"data: ")
    second = next(it)  # after sleep/appended line ("two")
    assert second.startswith(b"data: ")
