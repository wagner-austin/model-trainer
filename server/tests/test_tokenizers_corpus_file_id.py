from __future__ import annotations

from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizer_enqueue_with_corpus_file_id(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.tokenizer_orchestrator._redis = fake  # attribute exists on orchestrator

    captured: dict[str, object] = {}

    def _fake_enqueue_tokenizer(payload: dict[str, object]) -> str:
        nonlocal captured
        captured = payload
        return "job-1"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_tokenizer", _fake_enqueue_tokenizer)

    client = TestClient(app)
    body = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    r = client.post("/tokenizers/train", json=body)
    assert r.status_code == 200
    assert captured.get("corpus_file_id") == "deadbeef"
