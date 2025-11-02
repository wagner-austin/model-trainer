from __future__ import annotations

import os
from pathlib import Path

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backends.gpt2 import GPT2TrainConfig, prepare_gpt2
from model_trainer.core.services.model.gpt2_backend_impl import GPT2BackendImpl
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend


def test_gpt2_prepare_from_artifact(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    settings = Settings()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")

    tok_id = "tok-prep"
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
    prepared = prepare_gpt2(cfg, settings)
    assert prepared.max_seq_len == 16
    assert isinstance(prepared.eos_id, int)
    assert isinstance(prepared.pad_id, int)


def test_gpt2_backend_impl_end_to_end(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    settings = Settings()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")

    tok_id = "tok-backend"
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
    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))

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

    builder = LocalTextDatasetBuilder()
    backend = GPT2BackendImpl(dataset_builder=builder)
    prepared = backend.prepare(ModelTrainConfig(**cfg.model_dump()), settings, tokenizer=tok_handle)

    # Run a tiny train and save
    def _hb(_: float) -> None:  # pragma: no cover - trivial
        pass

    def _cancelled() -> bool:  # pragma: no cover - trivial
        return False

    res = backend.train(
        ModelTrainConfig(**cfg.model_dump()),
        settings,
        run_id="run-backend",
        heartbeat=_hb,
        cancelled=_cancelled,
        prepared=prepared,
    )
    assert res.loss >= 0.0
    _ = backend.save(prepared, str(artifacts / "models" / "run-backend"))

    # Evaluate path via backend adapter
    eval_res = backend.evaluate(
        run_id="run-backend", cfg=ModelTrainConfig(**cfg.model_dump()), settings=settings
    )
    assert eval_res.loss >= 0.0
