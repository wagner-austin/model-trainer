from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import training_worker as tw


def test_training_worker_failed_event_publish_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fake redis client that behaves normally
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(tw, "_redis_client", lambda: fake)

    # Force container creation to raise so we enter exception handler
    class _C:
        @staticmethod
        def from_settings(_: object) -> object:
            raise RuntimeError("boom during container creation")

    monkeypatch.setattr(tw, "ServiceContainer", _C)

    # Make _emit_failed_event raise to trigger the inner except branch (315-316)
    def _raise_failed_event(*_: object, **__: object) -> None:
        raise ValueError("publish fail")

    monkeypatch.setattr(tw, "_emit_failed_event", _raise_failed_event)

    payload: TrainJobPayload = {
        "run_id": "run-err",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": "tok",
            "corpus_path": str(tmp_path),
        },
    }

    with pytest.raises(RuntimeError):
        tw.process_train_job(payload)
    # Status is failed and message is set
    assert fake.get("runs:status:run-err") == "failed"
    assert isinstance(fake.get("runs:msg:run-err"), str)
