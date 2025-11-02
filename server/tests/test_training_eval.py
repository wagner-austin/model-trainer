from __future__ import annotations

import os
from pathlib import Path

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backends.gpt2 import (
    GPT2TrainConfig,
    evaluate_gpt2,
    prepare_gpt2_with_handle,
    train_prepared_gpt2,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from pydantic import BaseModel


def test_training_and_eval_tiny(tmp_path: Path, monkeypatch: object) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    settings = Settings()

    # Train tokenizer
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

    # Prepare tiny model
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

    def _hb(_: float) -> None:
        pass

    def _cancelled() -> bool:
        return False

    builder = LocalTextDatasetBuilder()
    # Load tokenizer handle and prepare model

    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))
    prepared = prepare_gpt2_with_handle(tok_handle, cfg)
    res = train_prepared_gpt2(
        prepared,
        cfg,
        settings,
        run_id="run-test",
        redis_hb=_hb,
        cancelled=_cancelled,
    )
    assert res.loss >= 0.0
    # Manifest written
    manifest = artifacts / "models" / "run-test" / "manifest.json"
    assert manifest.exists()

    class _Manifest(BaseModel):
        run_id: str
        epochs: int
        batch_size: int
        max_seq_len: int
        steps: int
        loss: float
        tokenizer_id: str
        corpus_path: str
        optimizer: str
        seed: int
        versions: dict[str, str]
        system: dict[str, object]
        git_commit: str | None = None

    text = manifest.read_text(encoding="utf-8")
    m = _Manifest.model_validate_json(text)
    assert m.tokenizer_id == tok_id

    # Eval metrics
    eval_res = evaluate_gpt2(run_id="run-test", cfg=cfg, settings=settings, dataset_builder=builder)
    assert eval_res.loss >= 0.0
    metrics_path = artifacts / "models" / "run-test" / "eval" / "metrics.json"
    assert metrics_path.exists()
