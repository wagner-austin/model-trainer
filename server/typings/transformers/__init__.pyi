from torch import Tensor
from torch.nn import Module

class GPT2Config:
    vocab_size: int
    n_positions: int
    n_ctx: int
    n_embd: int
    n_layer: int
    n_head: int
    bos_token_id: int
    eos_token_id: int
    def __init__(
        self: GPT2Config,
        *,
        vocab_size: int,
        n_positions: int,
        n_ctx: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> None: ...

class _ModelOutput:
    loss: Tensor

class GPT2LMHeadModel(Module):
    config: GPT2Config
    def __init__(self: GPT2LMHeadModel, config: GPT2Config) -> None: ...
    def save_pretrained(self: GPT2LMHeadModel, save_directory: str) -> None: ...
    @classmethod
    def from_pretrained(cls: type[GPT2LMHeadModel], save_directory: str) -> GPT2LMHeadModel: ...
    def forward(
        self: GPT2LMHeadModel, *, input_ids: Tensor, labels: Tensor | None = ...
    ) -> _ModelOutput: ...
