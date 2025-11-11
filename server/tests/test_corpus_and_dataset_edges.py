from __future__ import annotations

from pathlib import Path

import pytest
from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.services.data.corpus import list_text_files, sample_lines
from model_trainer.core.services.training.dataset_builder import CausalLMDataset, split_corpus_files


def test_list_text_files_single_file(tmp_path: Path) -> None:
    fp = tmp_path / "a.txt"
    fp.write_text("x", encoding="utf-8")
    out = list_text_files(str(fp))
    assert out == [str(fp)]


def test_sample_lines_zero_k(tmp_path: Path) -> None:
    # Build a corpus file with some lines
    fp = tmp_path / "b.txt"
    fp.write_text("a\n\n b \n", encoding="utf-8")
    out = sample_lines([str(fp)], 0, seed=1)
    assert out == []


def test_split_corpus_no_files_raises(tmp_path: Path) -> None:
    cfg = DatasetConfig(corpus_path=str(tmp_path), holdout_fraction=0.5)
    with pytest.raises(RuntimeError):
        split_corpus_files(cfg)


def test_dataset_len_zero_on_empty(tmp_path: Path) -> None:
    # Empty file yields no ids
    fp = tmp_path / "c.txt"
    fp.write_text("\n\n", encoding="utf-8")

    class _Tok:
        class _Enc:
            def __init__(self: _Tok._Enc, ids: list[int]) -> None:
                self._ids = ids

            @property
            def ids(self: _Tok._Enc) -> list[int]:
                return self._ids

        def encode(self: _Tok, text: str) -> _Tok._Enc:
            return _Tok._Enc([])

        def token_to_id(self: _Tok, token: str) -> int | None:
            return 0

        def get_vocab_size(self: _Tok) -> int:
            return 1

    ds = CausalLMDataset(files=[str(fp)], tokenizer=_Tok(), max_len=8, eos_id=1, pad_id=0)
    assert len(ds) == 0
