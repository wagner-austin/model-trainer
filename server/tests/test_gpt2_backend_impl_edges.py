from __future__ import annotations

import os
from pathlib import Path

import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backends.gpt2 import GPT2TrainConfig
from model_trainer.core.services.model.gpt2_backend_impl import GPT2BackendImpl
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend


def test_gpt2_backend_impl_name_and_type_errors(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    settings = Settings()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\n", encoding="utf-8")

    tok_id = "tok-bpe"
    out_dir = artifacts / "tokenizers" / tok_id
    # Train real tokenizer
    cfg_tok = {
        "method": "bpe",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_path": str(corpus),
        "holdout_fraction": 0.1,
        "seed": 42,
        "out_dir": str(out_dir),
    }
    from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig

    _ = BPEBackend().train(TokenizerTrainConfig(**cfg_tok))
    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))

    cfg = GPT2TrainConfig(
        model_family="gpt2",
        model_size="s",
        max_seq_len=8,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        tokenizer_id=tok_id,
        corpus_path=str(corpus),
    )
    backend = GPT2BackendImpl(LocalTextDatasetBuilder())
    assert backend.name() == "gpt2"
    # Prepare and save works for proper type
    prepared = backend.prepare(ModelTrainConfig(**cfg.model_dump()), settings, tokenizer=tok_handle)
    _ = backend.save(prepared, str(artifacts / "models" / "run-x"))
    # Saving wrong type raises TypeError
    with pytest.raises(TypeError):
        _ = backend.save(object(), str(artifacts / "models" / "run-y"))
