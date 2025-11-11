from __future__ import annotations

import time
from typing import Protocol

from redis.exceptions import RedisError


class _RedisGetProto(Protocol):
    def get(self: _RedisGetProto, key: str) -> str | None: ...


class _RedisSetProto(Protocol):
    def set(self: _RedisSetProto, key: str, value: str) -> object: ...


def get_with_retry(client: _RedisGetProto, key: str, *, attempts: int = 3) -> str | None:
    delay = 0.01
    for i in range(attempts):
        try:
            return client.get(key)
        except RedisError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None


def set_with_retry(client: _RedisSetProto, key: str, value: str, *, attempts: int = 3) -> None:
    delay = 0.01
    for i in range(attempts):
        try:
            client.set(key, value)
            return
        except RedisError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
