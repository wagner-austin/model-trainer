from __future__ import annotations

from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.data import corpus_fetcher as cf


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

    # Stub CorpusFetcher.fetch to return a path under tmp_path
    class _Stub:
        def __init__(self: _Stub, *_: object, **__: object) -> None:
            return

        def fetch(self: _Stub, fid: str) -> Path:  # pragma: no cover - trivial branch
            p = tmp_path / f"{fid}.txt"
            p.write_text("hello", encoding="utf-8")
            return p

    monkeypatch.setattr(container.rq_enqueuer, "enqueue_tokenizer", _fake_enqueue_tokenizer)
    monkeypatch.setattr(cf, "CorpusFetcher", _Stub)

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
    assert isinstance(captured.get("corpus_path"), str)
    assert str(tmp_path) in str(captured["corpus_path"])
