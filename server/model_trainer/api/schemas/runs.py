from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    model_family: Literal["gpt2", "llama", "qwen"] = Field(default="gpt2")
    model_size: str = Field(default="small")
    max_seq_len: int = Field(default=512, ge=8)
    num_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=5e-4, gt=0)
    corpus_path: str = Field(description="Path to text corpus root or file")
    tokenizer_id: str = Field(description="Tokenizer artifact ID to use")


class TrainResponse(BaseModel):
    run_id: str
    job_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    last_heartbeat_ts: float | None = None
    message: str | None = None


class EvaluateRequest(BaseModel):
    split: Literal["validation", "test"] = Field(default="validation")
    path_override: str | None = None


class EvaluateResponse(BaseModel):
    run_id: str
    split: str
    status: Literal["queued", "running", "completed", "failed"]
    loss: float | None = None
    perplexity: float | None = None
    artifact_path: str | None = None

