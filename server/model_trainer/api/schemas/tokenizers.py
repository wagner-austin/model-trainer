from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TokenizerTrainRequest(BaseModel):
    method: Annotated[Literal["bpe", "sentencepiece"], Field(default="bpe")]
    vocab_size: Annotated[int, Field(default=32000, ge=128)]
    min_frequency: Annotated[int, Field(default=2, ge=1)]
    corpus_path: Annotated[str | None, Field(default=None, description="Corpus path")]
    corpus_file_id: Annotated[
        str | None, Field(default=None, description="data-bank-api file ID for corpus")
    ]
    holdout_fraction: Annotated[float, Field(default=0.01, ge=0.0, le=0.5)]
    seed: Annotated[int, Field(default=42)]

    model_config = {"extra": "forbid", "validate_assignment": True}


class TokenizerTrainResponse(BaseModel):
    tokenizer_id: str
    artifact_path: str
    coverage: float | None = None
    oov_rate: float | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class TokenizerInfoResponse(BaseModel):
    tokenizer_id: str
    artifact_path: str
    status: str
    coverage: float | None = None
    oov_rate: float | None = None
    token_count: int | None = None
    char_coverage: float | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}
