from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class GPT2TrainConfig(BaseModel):
    model_family: Literal["gpt2"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str
    corpus_path: str

    model_config = {"extra": "forbid", "validate_assignment": True}
