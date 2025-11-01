from __future__ import annotations

import time

import redis
from redis.exceptions import RedisError


def get_with_retry(client: redis.Redis[str], key: str, *, attempts: int = 3) -> str | None:
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


def set_with_retry(client: redis.Redis[str], key: str, value: str, *, attempts: int = 3) -> None:
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
