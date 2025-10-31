from __future__ import annotations

from pathlib import Path

from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from pydantic import BaseModel


def test_bpe_trains_and_writes_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    out_dir = artifacts / "tokenizers" / "tok-test"

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    stats = BPEBackend().train(cfg)
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert 0.0 <= stats.coverage <= 1.0

    class _Cfg(BaseModel):
        vocab_size: int
        min_frequency: int
        holdout_fraction: float
        seed: int
        special_tokens: list[str]

    class _Stats(BaseModel):
        coverage: float
        oov_rate: float
        token_count: int
        char_coverage: float

    class _Manifest(BaseModel):
        created_at: int
        config: _Cfg
        stats: _Stats

    text = (out_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = _Manifest.model_validate_json(text)
    assert isinstance(manifest.config.vocab_size, int)
    assert 0.0 <= manifest.stats.coverage <= 1.0
    assert 0.0 <= manifest.stats.char_coverage <= 1.0
