from __future__ import annotations

from pathlib import Path
from typing import Protocol

from model_trainer.core.services.training.dataset_builder import _TokenizerProto

from .types import GPT2Prepared, TokenHandleProto


class _TokWrapper(_TokenizerProto):
    class _Enc:
        def __init__(self: _TokWrapper._Enc, ids: list[int]) -> None:
            self._ids = ids

        @property
        def ids(self: _TokWrapper._Enc) -> list[int]:
            return self._ids

    def __init__(self: _TokWrapper, handle: TokenHandleProto) -> None:
        self._h = handle

    def encode(self: _TokWrapper, text: str) -> _TokWrapper._Enc:
        ids = self._h.encode(text)
        return _TokWrapper._Enc(ids)

    def token_to_id(self: _TokWrapper, token: str) -> int | None:
        return self._h.token_to_id(token)

    def get_vocab_size(self: _TokWrapper) -> int:
        return self._h.get_vocab_size()


class _TokenInfoProto(Protocol):
    def token_to_id(self: _TokenInfoProto, token: str) -> int | None: ...
    def get_vocab_size(self: _TokenInfoProto) -> int: ...


def token_ids(tokenizer: _TokenInfoProto) -> tuple[int, int, int]:
    eos_id_opt = tokenizer.token_to_id("[EOS]")
    eos_id = int(eos_id_opt) if eos_id_opt is not None else 0
    pad_id_opt = tokenizer.token_to_id("[PAD]")
    pad_id = int(pad_id_opt) if pad_id_opt is not None else 0
    vocab_size = int(tokenizer.get_vocab_size())
    return eos_id, pad_id, vocab_size


def load_tokenizer_for_dataset(tokenizer_path: str) -> _TokenizerProto:
    from tokenizers import Tokenizer  # typed via local stubs

    return Tokenizer.from_file(tokenizer_path)


def save_prepared_gpt2(prepared: GPT2Prepared, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prepared.model.save_pretrained(out_dir)


def load_prepared_gpt2_from_handle(artifact_path: str, tokenizer: TokenHandleProto) -> GPT2Prepared:
    from transformers import GPT2LMHeadModel  # typed via local stubs

    eos_id, pad_id, _ = token_ids(tokenizer)
    model = GPT2LMHeadModel.from_pretrained(artifact_path)
    max_seq_len: int = 512
    # Avoid broad exception handling by using getattr default
    attr: object | None = getattr(model.config, "n_positions", None)
    if isinstance(attr, int):
        max_seq_len = attr
    return GPT2Prepared(
        model=model,
        tokenizer_id="unknown",
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        tok_for_dataset=_TokWrapper(tokenizer),
    )
