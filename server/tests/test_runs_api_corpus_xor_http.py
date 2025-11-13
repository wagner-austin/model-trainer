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


def test_runs_train_missing_corpus_file_id_returns_422(tmp_path: Path) -> None:
    client = _mk_app(tmp_path)
    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        # Missing required corpus_file_id
        "tokenizer_id": "tok-1",
    }
    r = client.post("/runs/train", json=body)
    assert r.status_code == 422
    assert "corpus_file_id" in r.text and "missing" in r.text.lower()


def test_runs_train_extra_field_corpus_path_forbidden_returns_422(tmp_path: Path) -> None:
    client = _mk_app(tmp_path)
    body = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        "corpus_file_id": "deadbeef",
        "corpus_path": str(tmp_path / "corpus.txt"),
        "tokenizer_id": "tok-1",
    }
    r = client.post("/runs/train", json=body)
    assert r.status_code == 422
    assert "extra_forbidden" in r.text.lower() or "extra inputs" in r.text.lower()
