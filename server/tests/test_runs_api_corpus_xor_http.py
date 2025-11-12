from __future__ import annotations

from pathlib import Path

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def _mk_app(tmp: Path) -> TestClient:
    # Ensure artifacts and runs folders are within tmp
    (tmp / "artifacts").mkdir(exist_ok=True)
    (tmp / "runs").mkdir(exist_ok=True)
    s = Settings()
    s.app.artifacts_root = str(tmp / "artifacts")
    s.app.runs_root = str(tmp / "runs")
    app = create_app(s)

    # Swap out redis with fakeredis
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake
    return TestClient(app)


def test_runs_train_missing_both_corpus_fields_returns_400(tmp_path: Path) -> None:
    client = _mk_app(tmp_path)
    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        # Neither corpus_path nor corpus_file_id provided
        "tokenizer_id": "tok-1",
    }
    r = client.post("/runs/train", json=body)
    assert r.status_code == 400
    hdrs = {k.lower(): v for (k, v) in r.headers.items()}
    assert "config_invalid" in r.text.lower() and "x-request-id" in hdrs and hdrs["x-request-id"]


def test_runs_train_both_corpus_fields_returns_400(tmp_path: Path) -> None:
    client = _mk_app(tmp_path)
    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        "corpus_path": str(tmp_path / "corpus.txt"),
        "corpus_file_id": "deadbeef",
        "tokenizer_id": "tok-1",
    }
    r = client.post("/runs/train", json=body)
    assert r.status_code == 400
    hdrs2 = {k.lower(): v for (k, v) in r.headers.items()}
    assert "config_invalid" in r.text.lower() and "x-request-id" in hdrs2 and hdrs2["x-request-id"]
