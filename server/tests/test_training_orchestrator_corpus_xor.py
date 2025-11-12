from __future__ import annotations

import fakeredis
import pytest
from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


class _FakeEnq(RQEnqueuer):
    def __init__(self: _FakeEnq) -> None:
        super().__init__(redis_url="redis://x", settings=RQSettings("q", 1, 1, 1, 0, []))


def _orch() -> TrainingOrchestrator:
    fake = fakeredis.FakeRedis(decode_responses=True)
    return TrainingOrchestrator(settings=Settings(), redis_client=fake, enqueuer=_FakeEnq())


def test_enqueue_training_raises_when_neither_corpus_source_provided() -> None:
    orch = _orch()
    req = TrainRequest(
        model_family="gpt2",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        corpus_path=None,
        corpus_file_id=None,
        tokenizer_id="tok",
    )
    from model_trainer.core.errors.base import AppError, ErrorCode

    with pytest.raises(AppError) as ei:
        _ = orch.enqueue_training(req)
    assert ei.value.code == ErrorCode.CONFIG_INVALID


def test_enqueue_training_raises_when_both_corpus_sources_provided() -> None:
    orch = _orch()
    req = TrainRequest(
        model_family="gpt2",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        corpus_path="/path.txt",
        corpus_file_id="deadbeef",
        tokenizer_id="tok",
    )
    from model_trainer.core.errors.base import AppError, ErrorCode

    with pytest.raises(AppError) as ei:
        _ = orch.enqueue_training(req)
    assert ei.value.code == ErrorCode.CONFIG_INVALID
