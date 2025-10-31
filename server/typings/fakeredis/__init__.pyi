from __future__ import annotations

import redis as _redis

class FakeRedis(_redis.Redis[str]):
    def __init__(self: FakeRedis, decode_responses: bool = ...) -> None: ...
