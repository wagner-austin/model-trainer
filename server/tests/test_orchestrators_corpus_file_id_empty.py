from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest
from model_trainer.api.schemas.runs import TrainRequest
from model_trainer.api.schemas.tokenizers import TokenizerTrainRequest
from model_trainer.core.config.settings import Settings
from model_trainer.core.errors.base import AppError, ErrorCode
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator


def _enqueuer() -> RQEnqueuer:
    return RQEnqueuer(
        redis_url="redis://localhost:6379/0",
        settings=RQSettings(
            queue_name="q",
            job_timeout_sec=60,
            result_ttl_sec=60,
            failure_ttl_sec=60,
            retry_max=0,
            retry_intervals=[],
        ),
    )


def test_tokenizer_orchestrator_rejects_empty_corpus_file_id(tmp_path: Path) -> None:
    s = Settings()
    # point artifacts root to tmp to avoid unrelated writes
    s.app.artifacts_root = str(tmp_path / "artifacts")
    r = fakeredis.FakeRedis(decode_responses=True)
    orch = TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=_enqueuer())
    # Provide whitespace corpus_file_id which passes schema min_length but fails after strip
    req = TokenizerTrainRequest(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_file_id=" ",
        holdout_fraction=0.1,
        seed=1,
    )
    with pytest.raises(AppError) as ei:
        _ = orch.enqueue_training(req)
    assert ei.value.code == ErrorCode.CONFIG_INVALID


def test_training_orchestrator_rejects_empty_corpus_file_id(tmp_path: Path) -> None:
    s = Settings()
    s.app.artifacts_root = str(tmp_path / "artifacts")
    r = fakeredis.FakeRedis(decode_responses=True)
    orch = TrainingOrchestrator(
        settings=s, redis_client=r, enqueuer=_enqueuer(), model_registry=None
    )
    req = TrainRequest(
        model_family="gpt2",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        corpus_file_id=" ",
        tokenizer_id="tok",
        user_id=0,
    )
    with pytest.raises(AppError) as ei2:
        _ = orch.enqueue_training(req)
    assert ei2.value.code == ErrorCode.CONFIG_INVALID
