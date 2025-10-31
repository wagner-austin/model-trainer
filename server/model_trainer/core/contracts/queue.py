from __future__ import annotations

from typing import Literal, TypedDict


class TrainRequestPayload(TypedDict):
    model_family: Literal["gpt2", "llama", "qwen"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    corpus_path: str
    tokenizer_id: str


class TrainJobPayload(TypedDict):
    run_id: str
    request: TrainRequestPayload


class EvalJobPayload(TypedDict):
    run_id: str
    split: str
    path_override: str | None


class TokenizerTrainPayload(TypedDict):
    tokenizer_id: str
    method: Literal["bpe", "sentencepiece"]
    vocab_size: int
    min_frequency: int
    corpus_path: str
    holdout_fraction: float
    seed: int
