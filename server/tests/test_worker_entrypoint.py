from __future__ import annotations

import os

import fakeredis
import pytest
import redis
from model_trainer.worker.rq_worker import main


def test_worker_execs_rq_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: env and fake redis that pings True
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    fake: fakeredis.FakeRedis = fakeredis.FakeRedis(decode_responses=False)

    def _from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        assert url == os.environ["REDIS_URL"]
        assert decode_responses is False
        return fake

    monkeypatch.setattr("model_trainer.worker.rq_worker.redis.from_url", _from_url)

    # Simulate CLI present
    def _which(_: str) -> str:
        return "/bin/rq"

    monkeypatch.setattr("model_trainer.worker.rq_worker._shutil.which", _which)

    class _Rec:
        def __init__(self: _Rec) -> None:
            self.bin: str = ""
            self.args: list[str] = []

    rec = _Rec()

    def _execvp(bin_path: str, args: list[str]) -> None:
        rec.bin = bin_path
        rec.args = args
        raise SystemExit(0)

    monkeypatch.setattr("model_trainer.worker.rq_worker._os.execvp", _execvp)

    # Act + Assert
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 0
    assert rec.bin == "/bin/rq"
    assert rec.args[:2] == ["rq", "worker"]
    # Default queue name comes from settings (training)
    assert "training" in rec.args
    assert "--with-scheduler" in rec.args


def test_worker_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    # Fake redis OK
    fake: fakeredis.FakeRedis = fakeredis.FakeRedis(decode_responses=False)

    def _from_url_ok(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("model_trainer.worker.rq_worker.redis.from_url", _from_url_ok)

    # No rq binary found
    def _which_none(_: str) -> str | None:
        return None

    monkeypatch.setattr("model_trainer.worker.rq_worker._shutil.which", _which_none)

    with pytest.raises(RuntimeError):
        main([])


def test_worker_redis_ping_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"

    class _Bad:
        def ping(self: _Bad) -> bool:
            raise redis.exceptions.RedisError("boom")

    def _from_url_bad(url: str, decode_responses: bool = False) -> _Bad:
        return _Bad()

    monkeypatch.setattr("model_trainer.worker.rq_worker.redis.from_url", _from_url_bad)

    with pytest.raises(RuntimeError):
        main([])
