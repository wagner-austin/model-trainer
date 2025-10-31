from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker.training_worker import (
    HEARTBEAT_KEY_PREFIX,
    STATUS_KEY_PREFIX,
    process_train_job,
)


def test_training_cancellation_with_redis(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Use fake redis
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.training_worker.redis.from_url",
        _fake_from_url,
    )

    # Prepare artifacts and tokenizer
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    _ = Settings()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Make corpus long enough to allow cancel
    (corpus / "a.txt").write_text(("hello world\n" * 200), encoding="utf-8")
    tok_id = "tok-test"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    run_id = "run-cancel"
    payload = {
        "run_id": run_id,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 5,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "corpus_path": str(corpus),
            "tokenizer_id": tok_id,
        },
    }

    # Run training in a thread and cancel shortly after start
    t = threading.Thread(target=process_train_job, args=(payload,))
    t.start()
    time.sleep(0.2)
    fake.set(f"runs:{run_id}:cancelled", "1")
    t.join()

    status = fake.get(f"{STATUS_KEY_PREFIX}{run_id}")
    assert status == "failed"  # cancelled leads to failure status
    hb = fake.get(f"{HEARTBEAT_KEY_PREFIX}{run_id}")
    assert hb is not None
