from __future__ import annotations

import os
from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job


def test_tokenizer_worker_bpe_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange artifacts and a tiny corpus
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nhello ai\n", encoding="utf-8")

    # Fake redis in worker
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("model_trainer.worker.tokenizer_worker.redis.from_url", _fake_from_url)

    # Execute BPE path to full completion (hits final logging extras)
    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-bpe",
        "method": "bpe",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_path": str(corpus),
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    process_tokenizer_train_job(payload)

    # Assert status and stats were stored (exercising end-of-function lines)
    assert fake.get("tokenizer:tok-bpe:status") == "completed"
    stats_json = fake.get("tokenizer:tok-bpe:stats")
    assert isinstance(stats_json, str) and "coverage" in stats_json
