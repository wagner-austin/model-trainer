from __future__ import annotations

from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.runs import RunStatusResponse, TrainResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer
from pydantic import BaseModel


def test_api_key_unauthorized_and_authorized(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SECURITY__API_KEY", "sekret")
    # Artifacts root for manifest/log paths
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    app = create_app(Settings())

    # Wire fake redis and stub enqueuer
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake

    def _fake_enqueue_train(payload: dict[str, object]) -> str:
        return "job-1"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_train", _fake_enqueue_train)

    # Stub CorpusFetcher to map file id to local corpus path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: object, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: object, file_id: str) -> Path:
            (tmp_path / "corpus").mkdir(exist_ok=True)
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
        "tokenizer_id": "tok-1",
        "user_id": 123,
    }
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("hello", encoding="utf-8")

    r1 = client.post("/runs/train", json=body)
    # Requires API key -> 401
    assert r1.status_code == 401

    class _Err(BaseModel):
        code: str
        message: str
        request_id: str

    parsed = _Err.model_validate_json(r1.text)
    assert parsed.code == "UNAUTHORIZED" and parsed.request_id

    r2 = client.post("/runs/train", json=body, headers={"X-Api-Key": "sekret"})
    assert r2.status_code == 200
    run_id = TrainResponse.model_validate_json(r2.text).run_id

    # Populate a status message and verify surface via GET /runs/{id}
    key_status = f"runs:status:{run_id}"
    key_msg = f"runs:msg:{run_id}"
    fake.set(key_status, "failed")
    fake.set(key_msg, "boom")

    r3 = client.get(f"/runs/{run_id}", headers={"X-Api-Key": "sekret"})
    assert r3.status_code == 200
    out = RunStatusResponse.model_validate_json(r3.text)
    assert out.status == "failed" and out.message == "boom"
