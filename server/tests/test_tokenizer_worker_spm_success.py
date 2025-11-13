from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainStats
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job


def test_tokenizer_worker_spm_success(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(_url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis.from_url",
        _fake_from_url,
    )

    # Force CLI available
    def _which(_name: str) -> str:
        return "spm"

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.shutil.which",
        _which,
    )

    # Stub the backend class imported inside the worker branch
    class _SPMBackend:
        def train(self: _SPMBackend, cfg: object) -> TokenizerTrainStats:
            return TokenizerTrainStats(
                coverage=0.9,
                oov_rate=0.1,
                token_count=10,
                char_coverage=0.8,
            )

    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.spm_backend.SentencePieceBackend",
        _SPMBackend,
        raising=True,
    )

    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    # Minimal corpus
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "a.txt").write_text("hi\n", encoding="utf-8")

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm-succ",
        "method": "sentencepiece",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_path": str(tmp_path / "c"),
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm-succ:status") == "completed"
    stats_json = fake.get("tokenizer:tok-spm-succ:stats")
    assert isinstance(stats_json, str) and "oov_rate" in stats_json
