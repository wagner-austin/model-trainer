from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.model.backends.gpt2 import (
    GPT2TrainConfig,
    prepare_gpt2_with_handle,
    train_prepared_gpt2,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker.training_worker import (
    EVAL_KEY_PREFIX,
    process_eval_job,
)
from pytest import MonkeyPatch


def test_eval_job_success(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Use fake redis
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("redis.from_url", _fake_from_url)

    # Prepare artifacts and tokenizer
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    _ = Settings()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")

    tok_id = "tok-eval"
    tok_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    # Train and persist a tiny model for run_id
    run_id = "run-eval"
    cfg = GPT2TrainConfig(
        model_family="gpt2",
        model_size="small",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        tokenizer_id=tok_id,
        corpus_path=str(corpus),
    )
    tok_handle = BPEBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = prepare_gpt2_with_handle(tok_handle, cfg)

    # heartbeat/cancel no-ops
    def _hb(_: float) -> None:  # pragma: no cover - trivial
        pass

    def _cancelled() -> bool:  # pragma: no cover - trivial
        return False

    _ = train_prepared_gpt2(
        prepared,
        cfg,
        Settings(),
        run_id=run_id,
        redis_hb=_hb,
        cancelled=_cancelled,
    )

    # Now process eval job using the worker entry
    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    process_eval_job(payload)
    raw = fake.get(f"{EVAL_KEY_PREFIX}{run_id}")
    assert raw is not None
    # Ensure status completed and metrics are present
    from pydantic import BaseModel

    class _Eval(BaseModel):
        status: str
        split: str
        loss: float | None = None
        ppl: float | None = None
        artifact: str | None = None

    out = _Eval.model_validate_json(raw)
    assert out.status == "completed"
    assert out.loss is not None
    assert out.ppl is not None


def test_eval_job_missing_manifest(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("redis.from_url", _fake_from_url)

    # Point artifacts to empty dir so manifest is absent
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    _ = Settings()

    run_id = "run-missing"
    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    process_eval_job(payload)
    raw = fake.get(f"{EVAL_KEY_PREFIX}{run_id}")
    assert raw is not None
    from pydantic import BaseModel

    class _Eval(BaseModel):
        status: str
        split: str

    out = _Eval.model_validate_json(raw)
    assert out.status == "failed"
