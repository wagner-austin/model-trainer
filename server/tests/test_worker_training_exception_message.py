from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import training_worker as tw


def test_training_worker_sets_status_message_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fake redis client
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(tw, "_redis_client", lambda: fake)

    # Ensure artifacts root has no tokenizer so worker fails with FileNotFoundError
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    # Build payload with nonexistent tokenizer
    payload: TrainJobPayload = {
        "run_id": "run-x",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0005,
            "corpus_path": str(tmp_path / "corpus"),
            "tokenizer_id": "tok-missing",
        },
    }

    # Ensure corpus dir exists
    (tmp_path / "corpus").mkdir()
    with pytest.raises(FileNotFoundError):
        tw.process_train_job(payload)  # raises due to missing tokenizer artifact

    # Status and message keys set
    assert fake.get("runs:status:run-x") == "failed"
    msg = fake.get("runs:msg:run-x")
    assert isinstance(msg, str) and msg
