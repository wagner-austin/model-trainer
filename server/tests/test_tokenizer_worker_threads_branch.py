from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job


def test_tokenizer_worker_respects_threads_env_branch(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Use fake redis
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(_url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis.from_url",
        _fake_from_url,
    )
    # Force threads branch (threads_cfg > 0)
    os.environ["APP__THREADS"] = "2"
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")

    # Prepare tiny corpus
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "a.txt").write_text("hi\n", encoding="utf-8")

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-thr",
        "method": "bpe",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_path": str(tmp_path / "c"),
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-thr:status") == "completed"
