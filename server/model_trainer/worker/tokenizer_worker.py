from __future__ import annotations

import logging
import os
from pathlib import Path

import redis

from ..core.contracts.queue import TokenizerTrainPayload
from ..core.contracts.tokenizer import TokenizerTrainConfig
from ..core.services.tokenizer.bpe_backend import BPEBackend


def _redis_client() -> redis.Redis[str]:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def process_tokenizer_train_job(payload: TokenizerTrainPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    tok_id = payload["tokenizer_id"]
    r.set(f"tokenizer:{tok_id}:status", "running")
    artifacts_root = os.getenv("APP__ARTIFACTS_ROOT", "/data/artifacts")
    out_dir = str(Path(artifacts_root) / "tokenizers" / tok_id)
    cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=payload["vocab_size"],
        min_frequency=payload["min_frequency"],
        corpus_path=payload["corpus_path"],
        holdout_fraction=payload["holdout_fraction"],
        seed=payload["seed"],
        out_dir=out_dir,
    )
    backend = BPEBackend()
    stats = backend.train(cfg)
    r.set(f"tokenizer:{tok_id}:status", "completed")
    r.set(f"tokenizer:{tok_id}:stats", stats.model_dump_json())
    log.info("Tokenizer training completed id=%s out=%s", tok_id, out_dir)
