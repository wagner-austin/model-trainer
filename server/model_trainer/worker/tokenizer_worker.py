from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import redis

from ..core.config.settings import Settings
from ..core.contracts.compute import LocalCPUProvider
from ..core.contracts.queue import TokenizerTrainPayload
from ..core.contracts.tokenizer import TokenizerTrainConfig, TokenizerTrainStats
from ..core.infra.paths import tokenizer_dir, tokenizer_logs_path
from ..core.logging.service import LoggingService
from ..core.services.data import corpus_fetcher as corpus_fetcher_mod
from ..core.services.tokenizer.bpe_backend import BPEBackend


def _redis_client() -> redis.Redis[str]:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def _has_spm_cli() -> bool:
    return all(shutil.which(x) is not None for x in ("spm_train", "spm_encode", "spm_decode"))


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
    # Disable tokenizer internal parallelism for stable CPU usage
    os.environ["TOKENIZERS_PARALLELISM"] = "1"

    out_dir = str(tokenizer_dir(settings, tok_id))
    # Attach per-tokenizer log file
    logsvc = LoggingService.create()
    tok_log_path = str(tokenizer_logs_path(settings, tok_id))
    tok_logger = logsvc.attach_run_file(
        path=tok_log_path, category="tokenizer", service="worker", tokenizer_id=tok_id
    )
    tok_logger.info(
        "Tokenizer job started",
        extra={
            "event": "tokenizer_started",
            "tokenizer_id": tok_id,
            "method": payload["method"],
            "vocab_size": int(payload["vocab_size"]),
        },
    )
    # Resolve corpus file id to local cache path
    fid = str(payload["corpus_file_id"]).strip()
    fetcher = corpus_fetcher_mod.CorpusFetcher(
        api_url=settings.app.data_bank_api_url,
        api_key=settings.app.data_bank_api_key,
        cache_dir=Path(settings.app.data_root) / "corpus_cache",
    )
    resolved_corpus = str(fetcher.fetch(fid))

    cfg = TokenizerTrainConfig(
        method=payload["method"],
        vocab_size=payload["vocab_size"],
        min_frequency=payload["min_frequency"],
        corpus_path=resolved_corpus,
        holdout_fraction=payload["holdout_fraction"],
        seed=payload["seed"],
        out_dir=out_dir,
        sample_max_lines=settings.app.tokenizer_sample_max_lines,
    )

    def _log_completed(stats: TokenizerTrainStats) -> None:
        tok_logger.info(
            "Tokenizer training completed",
            extra={
                "event": "tokenizer_completed",
                "tokenizer_id": tok_id,
                "vocab_size": int(payload["vocab_size"]),
                "coverage": float(stats.coverage),
                "oov_rate": float(stats.oov_rate),
                "token_count": int(stats.token_count),
                "char_coverage": float(stats.char_coverage),
            },
        )

    # Select backend by method and finalize per-branch
    if payload["method"] == "bpe":
        backend = BPEBackend()
        stats = backend.train(cfg)
        r.set(f"tokenizer:{tok_id}:status", "completed")
        r.set(f"tokenizer:{tok_id}:stats", stats.model_dump_json())
        _log_completed(stats)
        logsvc.close_run_file(path=tok_log_path)
        return
    # sentencepiece
    if _has_spm_cli():
        from ..core.services.tokenizer.spm_backend import SentencePieceBackend

        backend_spm = SentencePieceBackend()
        stats = backend_spm.train(cfg)
        r.set(f"tokenizer:{tok_id}:status", "completed")
        r.set(f"tokenizer:{tok_id}:stats", stats.model_dump_json())
        _log_completed(stats)
        logsvc.close_run_file(path=tok_log_path)
        return
    r.set(f"tokenizer:{tok_id}:status", "failed")
    log.error(
        "Unsupported tokenizer method: %s (SentencePiece CLI not available)",
        payload["method"],
    )
    tok_logger.info(
        "Tokenizer backend unavailable",
        extra={"event": "tokenizer_backend_unavailable", "tokenizer_id": tok_id},
    )
    logsvc.close_run_file(path=tok_log_path)
    return
