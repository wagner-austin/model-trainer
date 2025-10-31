from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TokenizerTrainRequest(BaseModel):
    method: Literal["bpe", "sentencepiece"] = Field(default="bpe")
    vocab_size: int = Field(default=32000, ge=128)
    min_frequency: int = Field(default=2, ge=1)
    corpus_path: str
    holdout_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    seed: int = Field(default=42)


class TokenizerTrainResponse(BaseModel):
    tokenizer_id: str
    artifact_path: str
    coverage: float | None = None
    oov_rate: float | None = None

