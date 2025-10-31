from __future__ import annotations

from typing import Literal, TypedDict


class LoggingExtra(TypedDict, total=False):
    event: str
    model_family: str
    model_size: str
    split: str
    kind: Literal["tokenizers", "models"]
    item_id: str
    count: int
    path: str
    status: str
    run_id: str
    loss: float
    perplexity: float
    steps: int
    tail: int
    method: str
    vocab_size: int
    tokenizer_id: str
    reason: str
