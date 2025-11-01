from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class TokenizerTrainConfig(BaseModel):
    method: str
    vocab_size: int
    min_frequency: int
    corpus_path: str
    holdout_fraction: float
    seed: int
    out_dir: str
    sample_max_lines: int | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class TokenizerTrainStats(BaseModel):
    coverage: float
    oov_rate: float
    token_count: int
    char_coverage: float

    model_config = {"extra": "forbid", "validate_assignment": True}


class TokenizerHandle(Protocol):
    def encode(self: TokenizerHandle, text: str) -> list[int]: ...
    def decode(self: TokenizerHandle, ids: list[int]) -> str: ...
    def token_to_id(self: TokenizerHandle, token: str) -> int | None: ...
    def get_vocab_size(self: TokenizerHandle) -> int: ...


class TokenizerBackend(Protocol):
    def name(self: TokenizerBackend) -> str: ...
    def train(self: TokenizerBackend, cfg: TokenizerTrainConfig) -> TokenizerTrainStats: ...
    def load(self: TokenizerBackend, artifact_path: str) -> TokenizerHandle: ...
    def encode(self: TokenizerBackend, handle: TokenizerHandle, text: str) -> list[int]: ...
    def decode(self: TokenizerBackend, handle: TokenizerHandle, ids: list[int]) -> str: ...
