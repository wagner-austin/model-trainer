from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel


class EvalCache(BaseModel):
    status: Literal["queued", "running", "completed", "failed"]
    split: str
    loss: float | None = None
    ppl: float | None = None
    artifact: str | None = None


class TrainingManifestVersions(TypedDict):
    torch: str
    transformers: str
    tokenizers: str
    datasets: str


class TrainingManifestSystem(TypedDict):
    cpu_count: int
    platform: str
    platform_release: str
    machine: str


class TrainingManifest(TypedDict):
    run_id: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    tokenizer_id: str
    corpus_path: str
    optimizer: str
    seed: int
    versions: TrainingManifestVersions
    system: TrainingManifestSystem
    git_commit: str | None
