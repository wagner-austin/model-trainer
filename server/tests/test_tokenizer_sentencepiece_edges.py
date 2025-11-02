from __future__ import annotations

from pathlib import Path

import pytest
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend
from pytest import MonkeyPatch


def test_sentencepiece_cli_unavailable_raises(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Force CLI not found
    import shutil as _shutil

    def _nope(_: str) -> None:
        return None

    monkeypatch.setattr(_shutil, "which", _nope)
    backend = SentencePieceBackend()
    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(tmp_path),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tmp_path / "tok"),
    )
    with pytest.raises(RuntimeError):
        backend.train(cfg)


def test_sentencepiece_inspect_missing_manifest(tmp_path: Path) -> None:
    backend = SentencePieceBackend()
    missing_dir = tmp_path / "nope"
    with pytest.raises(RuntimeError):
        backend.inspect(str(missing_dir))
