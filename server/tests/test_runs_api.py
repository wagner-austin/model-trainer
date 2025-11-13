from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.runs import RunStatusResponse, TrainResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_train_and_status_and_logs(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Ensure artifacts in temp
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    app = create_app(Settings())

    # Access container for patching
    container: ServiceContainer = app.state.container  # type narrowing

    # Swap redis client in orchestrator + container to fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake

    # Stub RQ enqueuer
    def _fake_enqueue_train(payload: dict[str, object]) -> str:
        return "job-test"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_train", _fake_enqueue_train)

    # Stub CorpusFetcher to map file id to local corpus path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: object, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: object, file_id: str) -> Path:
            return tmp_path / "corpus"

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

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
    # Create a minimal corpus to allow stats/logging paths to proceed
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("hello", encoding="utf-8")

    r = client.post("/runs/train", json=body)
    assert r.status_code == 200
    run_id = TrainResponse.model_validate_json(r.text).run_id

    # Status should be queued
    r2 = client.get(f"/runs/{run_id}")
    assert r2.status_code == 200
    assert RunStatusResponse.model_validate_json(r2.text).status == "queued"

    # Create a per-run log with known content and verify GET returns it
    log_dir = artifacts / "models" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "logs.jsonl").write_text('{"msg":"hello"}\n', encoding="utf-8")
    logs = client.get(f"/runs/{run_id}/logs", params={"tail": 10})
    assert logs.status_code == 200
    assert "hello" in logs.text
