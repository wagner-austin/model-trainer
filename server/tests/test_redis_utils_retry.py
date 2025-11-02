from __future__ import annotations

import fakeredis
import pytest
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry
from pytest import MonkeyPatch
from redis.exceptions import RedisError


def test_get_with_retry_succeeds_after_retry(monkeypatch: MonkeyPatch) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    state = {"calls": 0}

    def flaky_get(name: str) -> str | None:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RedisError("boom")
        return "v"

    monkeypatch.setattr(fake, "get", flaky_get)
    v = get_with_retry(fake, "k", attempts=2)
    assert v == "v"


def test_set_with_retry_exhausts_and_raises(monkeypatch: MonkeyPatch) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)

    def flaky_set(name: str, value: str, *args: object, **kwargs: object) -> bool | None:
        raise RedisError("boom")

    monkeypatch.setattr(fake, "set", flaky_set)

    with pytest.raises(RedisError):
        set_with_retry(fake, "k", "v", attempts=2)
