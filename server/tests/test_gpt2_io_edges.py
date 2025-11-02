from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from model_trainer.core.services.model.backends.gpt2.io import (
    load_prepared_gpt2_from_handle,
    save_prepared_gpt2,
    token_ids,
)
from model_trainer.core.services.model.backends.gpt2.types import GPT2Prepared
from pytest import MonkeyPatch
from torch import Tensor


class _FakeTokenInfo(Protocol):
    def token_to_id(self: _FakeTokenInfo, token: str) -> int | None: ...
    def get_vocab_size(self: _FakeTokenInfo) -> int: ...


class _Fwd:
    @property
    def loss(self: _Fwd) -> Tensor:
        return torch.tensor([0])[0]


class FakeModel:
    def train(self: FakeModel) -> None:
        pass

    def eval(self: FakeModel) -> None:
        pass

    def forward(self: FakeModel, *, input_ids: Tensor, labels: Tensor) -> _Fwd:
        return _Fwd()

    def parameters(self: FakeModel) -> list[Tensor]:
        return []

    def to(self: FakeModel, device: torch.device) -> object:
        return self

    def save_pretrained(self: FakeModel, out_dir: str) -> None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights.bin").write_bytes(b"ok")

    @property
    def config(self: FakeModel) -> object:
        class _Cfg:
            n_positions = 8

        return _Cfg()


class Enc:
    def __init__(self: Enc, ids: list[int]) -> None:
        self._ids = ids

    @property
    def ids(self: Enc) -> list[int]:
        return self._ids


class Tok:
    def encode(self: Tok, text: str) -> Enc:
        return Enc([1, 2, 3])

    def token_to_id(self: Tok, token: str) -> int | None:
        return 0

    def get_vocab_size(self: Tok) -> int:
        return 10


def test_token_ids_defaults_when_missing() -> None:
    class FakeTok:
        def token_to_id(self: FakeTok, token: str) -> int | None:
            return None

        def get_vocab_size(self: FakeTok) -> int:
            return 100

    eos_id, pad_id, vocab = token_ids(FakeTok())
    assert eos_id == 0
    assert pad_id == 0
    assert vocab == 100


def test_save_prepared_gpt2_writes(tmp_path: Path) -> None:
    prepared = GPT2Prepared(
        model=FakeModel(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=Tok(),
    )
    out_dir = str(tmp_path / "m")
    save_prepared_gpt2(prepared, out_dir)
    assert (tmp_path / "m" / "weights.bin").exists()


def test_load_prepared_gpt2_from_handle_uses_n_positions(monkeypatch: MonkeyPatch) -> None:
    class TokH:
        def encode(self: TokH, text: str) -> list[int]:
            return [1]

        def token_to_id(self: TokH, token: str) -> int | None:
            if token == "[EOS]":
                return 2
            if token == "[PAD]":
                return 0
            return None

        def get_vocab_size(self: TokH) -> int:
            return 16

    class _M:
        def __init__(self: _M) -> None:
            class _Cfg:
                n_positions = 64

            self.config = _Cfg()

        @classmethod
        def from_pretrained(cls: type[_M], path: str) -> _M:
            return cls()

    import transformers as _t

    monkeypatch.setattr(_t, "GPT2LMHeadModel", _M)
    prepared = load_prepared_gpt2_from_handle("/does/not/matter", TokH())
    assert prepared.max_seq_len == 64


def test_load_prepared_gpt2_from_handle_defaults_when_missing(monkeypatch: MonkeyPatch) -> None:
    class TokH:
        def encode(self: TokH, text: str) -> list[int]:
            return [1]

        def token_to_id(self: TokH, token: str) -> int | None:
            return None

        def get_vocab_size(self: TokH) -> int:
            return 16

    class _M2:
        def __init__(self: _M2) -> None:
            class _Cfg:
                pass

            self.config = _Cfg()

        @classmethod
        def from_pretrained(cls: type[_M2], path: str) -> _M2:
            return cls()

    import transformers as _t

    monkeypatch.setattr(_t, "GPT2LMHeadModel", _M2)
    prepared = load_prepared_gpt2_from_handle("/no/file", TokH())
    assert prepared.max_seq_len == 512
