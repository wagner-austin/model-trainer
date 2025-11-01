from __future__ import annotations

import logging
import os
import shutil

import redis

from ..core.config.settings import Settings
from ..core.contracts.compute import LocalCPUProvider
from ..core.contracts.queue import TokenizerTrainPayload
from ..core.contracts.tokenizer import TokenizerTrainConfig
from ..core.infra.paths import tokenizer_dir
from ..core.services.tokenizer.bpe_backend import BPEBackend


def _redis_client() -> redis.Redis[str]:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def process_tokenizer_train_job(payload: TokenizerTrainPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    tok_id = payload["tokenizer_id"]
    r.set(f"tokenizer:{tok_id}:status", "running")
    settings = Settings()
    # Apply local CPU compute environment
    threads_cfg = settings.app.threads
    threads = threads_cfg if threads_cfg and threads_cfg > 0 else max(1, int(os.cpu_count() or 1))
    env = LocalCPUProvider(threads_count=threads).env()
    for k, v in env.items():
        os.environ[k] = v

    out_dir = str(tokenizer_dir(settings, tok_id))
    cfg = TokenizerTrainConfig(
        method=payload["method"],
        vocab_size=payload["vocab_size"],
        min_frequency=payload["min_frequency"],
        corpus_path=payload["corpus_path"],
        holdout_fraction=payload["holdout_fraction"],
        seed=payload["seed"],
        out_dir=out_dir,
        sample_max_lines=settings.app.tokenizer_sample_max_lines,
    )
    # Select backend by method
    if payload["method"] == "bpe":
        backend = BPEBackend()
        stats = backend.train(cfg)
    elif payload["method"] == "sentencepiece":
        if all(shutil.which(x) is not None for x in ("spm_train", "spm_encode", "spm_decode")):
            from ..core.services.tokenizer.spm_backend import SentencePieceBackend

            backend_spm = SentencePieceBackend()
            stats = backend_spm.train(cfg)
        else:
            r.set(f"tokenizer:{tok_id}:status", "failed")
            log.error(
                "Unsupported tokenizer method: %s (SentencePiece CLI not available)",
                payload["method"],
            )
            return
    else:
        r.set(f"tokenizer:{tok_id}:status", "failed")
        log.error("Unsupported tokenizer method: %s", payload["method"])
        return
    r.set(f"tokenizer:{tok_id}:status", "completed")
    r.set(f"tokenizer:{tok_id}:stats", stats.model_dump_json())
    log.info("Tokenizer training completed id=%s out=%s", tok_id, out_dir)
