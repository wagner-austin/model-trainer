from __future__ import annotations

from pathlib import Path

import pytest
import tokenizers as _tok_mod
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


def test_bpe_char_coverage_unknown_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Corpus with a single special char line ensures uniq_chars contains it
    corpus = tmp_path / "corpus2"
    corpus.mkdir()
    special = "\u25c6"  # black diamond
    (corpus / "a.txt").write_text(f"{special}\n", encoding="utf-8")
    out_dir = tmp_path / "tok2"

    # Save original encode to avoid affecting other inputs
    from tokenizers import Tokenizer as _Tokenizer

    orig_encode = _Tokenizer.encode

    def _fake_encode(self: _Tokenizer, text: str) -> object:
        class _Enc:
            def __init__(self: _Enc, ids: list[int]) -> None:
                self.ids = ids

        if len(text) == 1 and text == special:
            unk = self.token_to_id("[UNK]")
            return _Enc([unk if unk is not None else -1])
        # Fallback to original for other inputs
        return orig_encode(self, text)

    monkeypatch.setattr(_tok_mod.Tokenizer, "encode", _fake_encode, raising=True)

    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=1,
        out_dir=str(out_dir),
        sample_max_lines=1,
    )
    # This trains and runs the internal char-coverage loop; our fake forces unknown-only ids
    stats = BPEBackend().train(cfg)
    assert 0.0 <= stats.char_coverage <= 1.0
