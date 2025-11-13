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


def test_train_request_missing_corpus_file_id_raises_validation_error() -> None:
    with pytest.raises(Exception):
        _ = TrainRequest(
            model_family="gpt2",
            model_size="s",
            max_seq_len=16,
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            tokenizer_id="tok",
        )


def test_train_request_extra_corpus_path_forbidden() -> None:
    with pytest.raises(Exception):
        _ = TrainRequest(
            model_family="gpt2",
            model_size="s",
            max_seq_len=16,
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            corpus_file_id="deadbeef",
            corpus_path="/path.txt",
            tokenizer_id="tok",
        )
