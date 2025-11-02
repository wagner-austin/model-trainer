from __future__ import annotations

from pathlib import Path

import pytest
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend


def test_bpe_train_no_files_raises(tmp_path: Path) -> None:
    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(tmp_path / "empty"),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tmp_path / "tok"),
    )
    with pytest.raises(RuntimeError):
        _ = BPEBackend().train(cfg)


def test_bpe_inspect_missing_manifest(tmp_path: Path) -> None:
    backend = BPEBackend()
    with pytest.raises(RuntimeError):
        _ = backend.inspect(str(tmp_path / "tok"))


def test_bpe_encode_decode_roundtrip(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\n", encoding="utf-8")
    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tmp_path / "tok"),
    )
    _ = BPEBackend().train(cfg)
    handle = BPEBackend().load(str(tmp_path / "tok" / "tokenizer.json"))
    ids = BPEBackend().encode(handle, "hello")
    s = BPEBackend().decode(handle, ids)
    assert isinstance(ids, list)
    assert isinstance(s, str)
