from __future__ import annotations

import fakeredis
import pytest
from model_trainer.api.schemas.runs import EvaluateRequest, TrainRequest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import EvalJobPayload, TrainJobPayload
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


class _FakeEnq(RQEnqueuer):
    def __init__(self: _FakeEnq) -> None:
        super().__init__(redis_url="redis://x", settings=RQSettings("q", 1, 1, 1, 0, []))
        self.last: tuple[str, TrainJobPayload | EvalJobPayload] | None = None

    def enqueue_train(self: _FakeEnq, payload: TrainJobPayload) -> str:
        self.last = ("train", payload)
        return "job-train"

    def enqueue_eval(self: _FakeEnq, payload: EvalJobPayload) -> str:
        self.last = ("eval", payload)
        return "job-eval"


def test_orchestrator_unsupported_model_raises() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    reg = ModelRegistry(backends={})
    orch = TrainingOrchestrator(
        settings=Settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=reg,
    )
    req = TrainRequest(
        model_family="llama",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        corpus_file_id="deadbeef",
        tokenizer_id="tok",
    )
    from model_trainer.core.errors.base import AppError

    with pytest.raises(AppError):
        _ = orch.enqueue_training(req)


def test_orchestrator_status_missing() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    orch = TrainingOrchestrator(
        settings=Settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    from model_trainer.core.errors.base import AppError

    with pytest.raises(AppError):
        _ = orch.get_status("no-run")


def test_orchestrator_eval_missing_run_returns_failed() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    orch = TrainingOrchestrator(
        settings=Settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    out = orch.enqueue_evaluation("no-run", EvaluateRequest(split="validation"))
    assert out.status == "failed"


def test_orchestrator_eval_enqueues_and_sets_cache() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    # Mark run as present
    fake.set("runs:status:run-ok", "running")
    orch = TrainingOrchestrator(
        settings=Settings(),
        redis_client=fake,
        enqueuer=_FakeEnq(),
        model_registry=None,
    )
    out = orch.enqueue_evaluation("run-ok", EvaluateRequest(split="validation"))
    assert out.status == "queued"
    raw = fake.get("runs:eval:run-ok")
    assert raw is not None
