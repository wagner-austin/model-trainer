from __future__ import annotations

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry


def test_get_with_retry_succeeds_after_transient(monkeypatch: MonkeyPatch) -> None:
    from redis.exceptions import RedisError

    client = fakeredis.FakeRedis(decode_responses=True)
    client.set("a", "b")
    orig_get = client.get
    calls = {"n": 0}

    def flaky_get(name: str) -> str | None:  # signature compatible for decode_responses=True
        if calls["n"] == 0:
            calls["n"] += 1
            raise RedisError("transient")
        return orig_get(name)

    monkeypatch.setattr(client, "get", flaky_get, raising=True)
    val = get_with_retry(client, "a")
    assert val == "b"


def test_set_with_retry_succeeds_after_transient(monkeypatch: MonkeyPatch) -> None:
    from redis.exceptions import RedisError

    client = fakeredis.FakeRedis(decode_responses=True)
    orig_set = client.set
    calls = {"n": 0}

    def flaky_set(name: str, value: str) -> bool | None:  # signature compatible
        if calls["n"] == 0:
            calls["n"] += 1
            raise RedisError("transient")
        return orig_set(name, value)

    monkeypatch.setattr(client, "set", flaky_set, raising=True)
    set_with_retry(client, "a", "b")
    assert client.get("a") == "b"
