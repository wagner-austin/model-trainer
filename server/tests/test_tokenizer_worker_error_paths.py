from __future__ import annotations

from pathlib import Path

import fakeredis
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job
from pytest import MonkeyPatch


def _setup_fake(monkeypatch: MonkeyPatch) -> fakeredis.FakeRedis:
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("redis.from_url", _fake_from_url)
    return fake


# Unknown method path is intentionally not tested to keep strict typing intact.


def test_tokenizer_worker_sentencepiece_unavailable_sets_failed(monkeypatch: MonkeyPatch) -> None:
    fake = _setup_fake(monkeypatch)
    # Ensure CLI not found
    import shutil as _shutil

    def _which(name: str) -> str | None:
        return None

    monkeypatch.setattr(_shutil, "which", _which)
    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm",
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    # Stub fetcher to return a dummy path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return Path("/data")

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm:status") == "failed"
