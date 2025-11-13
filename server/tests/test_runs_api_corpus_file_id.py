from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_train_with_corpus_file_id(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Ensure artifacts in temp
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    app = create_app(Settings())

    container: ServiceContainer = app.state.container  # type narrowing

    # Swap redis to fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake

    # Stub RQ to capture payload
    from model_trainer.core.contracts.queue import TrainJobPayload, TrainRequestPayload

    captured: list[TrainJobPayload] = []

    def _fake_enqueue_train(payload: TrainJobPayload) -> str:
        captured.append(payload)
        return "job-cfid"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_train", _fake_enqueue_train)

    client = TestClient(app)

    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0005,
        "corpus_file_id": "deadbeef",
        "tokenizer_id": "tok-abc",
    }

    r = client.post("/runs/train", json=body)
    assert r.status_code == 200
    assert captured, "payload should be captured"
    req: TrainRequestPayload = captured[0]["request"]
    assert req["corpus_file_id"] == "deadbeef"
