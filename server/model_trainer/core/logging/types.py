from __future__ import annotations

from typing import Literal, TypedDict


class LoggingExtra(TypedDict, total=False):
    # Core logging context
    category: str
    service: str
    event: str
    run_id: str
    tokenizer_id: str
    error_code: str
    # API and orchestrator fields
    model_family: str
    model_size: str
    split: str
    kind: Literal["tokenizers", "models"]
    item_id: str
    count: int
    path: str
    status: str
    loss: float
    perplexity: float
    steps: int
    tail: int
    method: str
    vocab_size: int
    reason: str
    # Corpus fetcher fields
    file_id: str
    api_url: str
    url: str
    expected_size: int
    actual_size: int
    size: int
    resume_from: int
    elapsed_seconds: float
