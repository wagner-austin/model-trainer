from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    model_family: Annotated[Literal["gpt2", "llama", "qwen"], Field(default="gpt2")]
    model_size: Annotated[str, Field(default="small")]
    max_seq_len: Annotated[int, Field(default=512, ge=8)]
    num_epochs: Annotated[int, Field(default=1, ge=1)]
    batch_size: Annotated[int, Field(default=4, ge=1)]
    learning_rate: Annotated[float, Field(default=5e-4, gt=0)]
    corpus_path: Annotated[str | None, Field(default=None, description="Corpus path")]
    corpus_file_id: Annotated[str | None, Field(default=None, description="data-bank-api file ID")]
    tokenizer_id: Annotated[str, Field(description="Tokenizer artifact ID to use")]
    user_id: Annotated[int, Field(default=0, description="Discord user ID for DM updates", ge=0)]

    # Cross-field validation for corpus source is enforced in orchestrator
    # to avoid mypy-plugin incompatibilities with pydantic model-level validators.

    model_config = {"extra": "forbid", "validate_assignment": True}


class TrainResponse(BaseModel):
    run_id: str
    job_id: str

    model_config = {"extra": "forbid", "validate_assignment": True}


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    last_heartbeat_ts: float | None = None
    message: str | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class EvaluateRequest(BaseModel):
    split: Annotated[Literal["validation", "test"], Field(default="validation")]
    path_override: str | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class EvaluateResponse(BaseModel):
    run_id: str
    split: str
    status: Literal["queued", "running", "completed", "failed"]
    loss: float | None = None
    perplexity: float | None = None
    artifact_path: str | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class CancelResponse(BaseModel):
    status: Literal["cancellation-requested"]

    model_config = {"extra": "forbid", "validate_assignment": True}
