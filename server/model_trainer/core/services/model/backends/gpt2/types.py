from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch
from torch import Tensor

from model_trainer.core.services.training.dataset_builder import _TokenizerProto


class ForwardOutProto(Protocol):
    @property
    def loss(self: ForwardOutProto) -> Tensor: ...


class LMModelProto(Protocol):
    def train(self: LMModelProto) -> None: ...
    def eval(self: LMModelProto) -> None: ...
    def forward(self: LMModelProto, *, input_ids: Tensor, labels: Tensor) -> ForwardOutProto: ...
    def parameters(self: LMModelProto) -> Iterable[Tensor]: ...
    def to(self: LMModelProto, device: torch.device) -> object: ...
    def save_pretrained(self: LMModelProto, out_dir: str) -> None: ...
    @property
    def config(self: LMModelProto) -> object: ...


class TokenHandleProto(Protocol):
    def encode(self: TokenHandleProto, text: str) -> list[int]: ...
    def token_to_id(self: TokenHandleProto, token: str) -> int | None: ...
    def get_vocab_size(self: TokenHandleProto) -> int: ...


class EncodingLike(Protocol):
    @property
    def ids(self: EncodingLike) -> list[int]: ...


class GPT2Prepared:
    def __init__(
        self: GPT2Prepared,
        *,
        model: LMModelProto,
        tokenizer_id: str,
        eos_id: int,
        pad_id: int,
        max_seq_len: int,
        tok_for_dataset: _TokenizerProto,
    ) -> None:
        self.model = model
        self.tokenizer_id = tokenizer_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_for_dataset = tok_for_dataset
