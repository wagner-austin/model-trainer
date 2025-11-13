from __future__ import annotations

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.tokenizers import TokenizerTrainResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_tokenizer_enqueue_passes_method(monkeypatch: MonkeyPatch) -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.tokenizer_orchestrator._redis = fake

    captured: dict[str, object] = {}

    def _fake_enqueue_tokenizer(payload: dict[str, object]) -> str:
        nonlocal captured
        captured = payload
        return "job-1"

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_tokenizer", _fake_enqueue_tokenizer)
    # Stub CorpusFetcher to map file id to a local path
    import tempfile
    from pathlib import Path

    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: object, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: object, file_id: str) -> Path:  # return a valid path
            return Path(tempfile.gettempdir())

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

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
    _ = TokenizerTrainResponse.model_validate_json(r.text)
    assert captured.get("method") == "bpe"
