from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry
from redis.exceptions import RedisError


def test_settings_toml_loader_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide APP_CONFIG_FILE but make open raise OSError to cover fallback
    cfg_path = tmp_path / "app.toml"
    os.environ["APP_CONFIG_FILE"] = str(cfg_path)

    def _fake_open(path: str, *a: object, **k: object) -> object:
        raise OSError("fail")

    monkeypatch.setattr(builtins, "open", _fake_open)
    _ = Settings()  # Should not raise


def test_settings_no_config_files_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings when no config files exist anywhere - covers line 34."""
    # Remove APP_CONFIG_FILE env var
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)

    # Make os.path.exists always return False so no config files are found
    def _fake_exists(path: str) -> bool:
        return False

    monkeypatch.setattr(os.path, "exists", _fake_exists)
    settings = Settings()
    # Should use defaults when no config file is found
    assert settings.app.artifacts_root  # Should have default value


class _Flaky:
    def __init__(self: _Flaky) -> None:
        self._set_calls = 0
        self._get_calls = 0
        self._store: dict[str, str] = {}

    def set(self: _Flaky, key: str, value: str) -> object:
        self._set_calls += 1
        if self._set_calls == 1:
            raise RedisError("transient")
        self._store[key] = value
        return True

    def get(self: _Flaky, key: str) -> str | None:
        self._get_calls += 1
        if self._get_calls == 1:
            raise RedisError("transient")
        return self._store.get(key)


def test_redis_utils_retry_branches() -> None:
    f = _Flaky()
    set_with_retry(f, "k", "v", attempts=2)
    out = get_with_retry(f, "k", attempts=2)
    assert out == "v"


class _AlwaysFails:
    """Mock Redis client that always raises RedisError."""

    def get(self: _AlwaysFails, key: str) -> str | None:
        raise RedisError("always fails")


def test_redis_utils_get_exhausts_retries_and_raises() -> None:
    """Test get_with_retry raises after exhausting all retry attempts - covers line 24."""
    client = _AlwaysFails()
    with pytest.raises(RedisError, match="always fails"):
        get_with_retry(client, "key", attempts=3)
