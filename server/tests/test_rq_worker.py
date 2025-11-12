from __future__ import annotations

import pytest
import redis
from model_trainer.core.config.settings import Settings
from model_trainer.worker import rq_worker as rqw


def test_env_str_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO", "abc")
    assert rqw._env_str("FOO") == "abc"


def test_env_str_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAR", "   ")
    with pytest.raises(RuntimeError):
        _ = rqw._env_str("BAR")


def test_queue_name_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Blank RQ queue name triggers validation error
    monkeypatch.setenv("RQ__QUEUE_NAME", "   ")
    cfg = Settings()
    with pytest.raises(RuntimeError):
        _ = rqw._queue_name_from_settings(cfg)


def test_run_happy_path_execvp_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide REDIS_URL and valid queue name via defaults
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    def _fake_which(_name: str) -> str | None:
        return "rq"

    def _fake_execvp(_file: str, _args: list[str]) -> None:
        # No-op to allow run() to return 0 and cover the final line
        return None

    # Avoid touching network or actual ping logic
    def _no_assert(_conn: redis.Redis[bytes]) -> None:
        return None

    monkeypatch.setattr(rqw, "_assert_redis_ok", _no_assert)
    monkeypatch.setattr("model_trainer.worker.rq_worker._shutil.which", _fake_which)
    monkeypatch.setattr("model_trainer.worker.rq_worker._os.execvp", _fake_execvp)

    out = rqw.run()
    assert out == 0


def test_run_missing_redis_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "")
    with pytest.raises(RuntimeError):
        _ = rqw.run()


def test_run_redis_ping_false_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    # Force any Redis instance to report unhealthy ping
    def _ping_false(self: redis.Redis[bytes], **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(redis.Redis, "ping", _ping_false)

    with pytest.raises(RuntimeError):
        _ = rqw.run()


def test_main_keyboard_interrupt_returns_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(rqw, "run", _boom)
    assert rqw.main([]) == 130
