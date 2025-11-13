from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job


def test_tokenizer_worker_uses_settings_artifacts_root(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Ensure worker uses our fake redis
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis.from_url",
        _fake_from_url,
    )

    # Configure artifacts root via Settings env
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-worker",
        "method": "bpe",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }

    # Prepare minimal corpus
    (tmp_path / "a.txt").write_text("hello world\n", encoding="utf-8")

    # Stub fetcher to return local tmp_path
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return tmp_path

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    process_tokenizer_train_job(payload)  # should complete and write under artifacts
    assert fake.get("tokenizer:tok-worker:status") == "completed"
    # Verify artifacts path used via Settings
    out_dir = artifacts / "tokenizers" / "tok-worker"
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "manifest.json").exists()
