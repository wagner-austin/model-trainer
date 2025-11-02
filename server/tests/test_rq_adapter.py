from __future__ import annotations

from types import ModuleType

import fakeredis
from model_trainer.core.contracts.queue import (
    EvalJobPayload,
    TokenizerTrainPayload,
    TrainJobPayload,
)
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from pytest import MonkeyPatch


class _FakeJob:
    def __init__(self: _FakeJob, job_id: str) -> None:
        self._id = job_id

    def get_id(self: _FakeJob) -> str:  # pragma: no cover - trivial
        return self._id


class _FakeQueue:
    last: tuple[str, dict[str, object], dict[str, object]] | None

    def __init__(self: _FakeQueue, name: str, connection: object) -> None:
        self.name = name
        self.connection = connection
        self.last = None

    def enqueue(
        self: _FakeQueue, path: str, payload: dict[str, object], **kwargs: object
    ) -> _FakeJob:
        # Record invocation for assertions
        self.last = (path, payload, kwargs)
        desc = kwargs.get("description", "job")
        job_id = f"id:{desc}"
        return _FakeJob(job_id)


class _FakeRetry:
    def __init__(self: _FakeRetry, max: int, interval: list[int]) -> None:
        self.max = max
        self.interval = interval


class _RQModule(ModuleType):
    Queue: type
    Retry: type


def _install_fake_rq(monkeypatch: MonkeyPatch, queue_ref: dict[str, _FakeQueue]) -> None:
    # Build a fake rq module exposing Queue and Retry with proper types
    mod = _RQModule("rq")
    fake_q = _FakeQueue("training", fakeredis.FakeRedis())
    queue_ref["q"] = fake_q

    class Queue:
        def __init__(self: Queue, name: str, connection: object) -> None:
            # Adapter for rq.Queue; returns shared fake queue
            self._q = fake_q

        def enqueue(
            self: Queue, path: str, payload: dict[str, object], **kwargs: object
        ) -> _FakeJob:
            return self._q.enqueue(path, payload, **kwargs)

    class Retry:
        def __init__(self: Retry, max: int, interval: list[int]) -> None:
            self.max = max
            self.interval = interval

    mod.Queue = Queue
    mod.Retry = Retry
    import sys

    sys.modules["rq"] = mod


def test_rq_enqueuer_methods(monkeypatch: MonkeyPatch) -> None:
    # Replace redis.from_url with fake to avoid real connections
    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fakeredis.FakeRedis()

    monkeypatch.setattr("redis.from_url", _fake_from_url)
    holder: dict[str, _FakeQueue] = {}
    _install_fake_rq(monkeypatch, holder)

    settings = RQSettings(
        queue_name="training",
        job_timeout_sec=60,
        result_ttl_sec=120,
        failure_ttl_sec=180,
        retry_max=3,
        retry_intervals=[1, 2, 3],
    )
    enq = RQEnqueuer(redis_url="redis://localhost/0", settings=settings)

    # Train job
    train_payload: TrainJobPayload = {
        "run_id": "run-1",
        "request": {
            "model_family": "gpt2",
            "model_size": "s",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "corpus_path": "/x",
            "tokenizer_id": "tok",
        },
    }
    jid = enq.enqueue_train(train_payload)
    assert jid.startswith("id:train:run-1")
    last = holder["q"].last
    assert last is not None
    path, payload, kwargs = last
    assert path == "model_trainer.worker.training_worker.process_train_job"
    assert payload["run_id"] == "run-1"
    assert kwargs["job_timeout"] == 60

    # Eval job
    holder["q"].last = None
    eval_payload: EvalJobPayload = {
        "run_id": "run-1",
        "split": "validation",
        "path_override": None,
    }
    jid2 = enq.enqueue_eval(eval_payload)
    assert jid2.startswith("id:eval:run-1:validation")
    last2 = holder["q"].last
    assert last2 is not None
    path2, payload2, kwargs2 = last2
    assert path2 == "model_trainer.worker.training_worker.process_eval_job"
    assert payload2["split"] == "validation"

    # Tokenizer job
    holder["q"].last = None
    tok_payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-1",
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_path": "/c",
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    jid3 = enq.enqueue_tokenizer(tok_payload)
    assert jid3.startswith("id:tokenizer:tok-1")
    last3 = holder["q"].last
    assert last3 is not None
    path3, payload3, kwargs3 = last3
    assert path3 == "model_trainer.worker.tokenizer_worker.process_tokenizer_train_job"
    assert payload3["method"] == "bpe"
