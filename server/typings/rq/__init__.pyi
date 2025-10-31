from __future__ import annotations

import redis as _redis

class Job:
    def get_id(self: Job) -> str: ...

class Retry:
    def __init__(self: Retry, *, max: int, interval: list[int] | int) -> None: ...

class Queue:
    def __init__(self: Queue, name: str, connection: _redis.Redis[str]) -> None: ...
    def enqueue(
        self: Queue,
        func: str,
        payload: dict[str, object],
        *,
        job_timeout: int,
        result_ttl: int,
        failure_ttl: int,
        retry: Retry,
        description: str,
    ) -> Job: ...

class Worker:
    @staticmethod
    def all(connection: _redis.Redis[str]) -> list[Worker]: ...
