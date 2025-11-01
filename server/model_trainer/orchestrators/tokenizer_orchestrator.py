from __future__ import annotations

import shutil
from dataclasses import dataclass

import redis

from ..api.schemas.tokenizers import TokenizerTrainRequest, TokenizerTrainResponse
from ..core.config.settings import Settings
from ..core.contracts.queue import TokenizerTrainPayload
from ..core.errors.base import AppError, ErrorCode
from ..core.infra.paths import tokenizer_logs_path
from ..core.logging.service import LoggingService
from ..core.services.queue.rq_adapter import RQEnqueuer


@dataclass
class TokenizerEnqueueOut:
    tokenizer_id: str
    job_id: str


class TokenizerOrchestrator:
    def __init__(
        self: TokenizerOrchestrator,
        *,
        settings: Settings,
        redis_client: redis.Redis[str],
        enqueuer: RQEnqueuer,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer
        self._logger = LoggingService.create().adapter(category="orchestrator", service="tokenizer")

    def enqueue_training(
        self: TokenizerOrchestrator, req: TokenizerTrainRequest
    ) -> TokenizerTrainResponse | None:
        # Early validation of backend availability
        if req.method == "sentencepiece" and not all(
            shutil.which(x) is not None for x in ("spm_train", "spm_encode", "spm_decode")
        ):
            self._logger.info(
                "tokenizer backend unavailable",
                extra={"event": "tokenizer_backend_unavailable", "method": req.method},
            )
            raise AppError(ErrorCode.CONFIG_INVALID, "sentencepiece backend unavailable")
        token_hash = abs(hash((req.method, req.vocab_size, req.corpus_path, req.seed))) % (10**10)
        tokenizer_id = f"tok-{token_hash:010d}"
        self._redis.set(f"tokenizer:{tokenizer_id}:status", "queued")
        payload: TokenizerTrainPayload = {
            "tokenizer_id": tokenizer_id,
            "method": req.method,
            "vocab_size": req.vocab_size,
            "min_frequency": req.min_frequency,
            "corpus_path": req.corpus_path,
            "holdout_fraction": req.holdout_fraction,
            "seed": req.seed,
        }
        # Per-tokenizer log file
        tok_log_path = str(tokenizer_logs_path(self._settings, tokenizer_id))
        logsvc = LoggingService.create()
        per_tok_logger = logsvc.attach_run_file(
            path=tok_log_path,
            category="tokenizer",
            service="orchestrator",
            tokenizer_id=tokenizer_id,
        )

        _ = self._enq.enqueue_tokenizer(payload)
        artifact_path = f"{self._settings.app.artifacts_root}/tokenizers/{tokenizer_id}"
        per_tok_logger.info(
            "tokenizer enqueued", extra={"event": "enqueued", "tokenizer_id": tokenizer_id}
        )
        logsvc.close_run_file(path=tok_log_path)
        return TokenizerTrainResponse(tokenizer_id=tokenizer_id, artifact_path=artifact_path)
