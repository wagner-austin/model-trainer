from __future__ import annotations

from dataclasses import dataclass

import redis

from ..api.schemas.tokenizers import TokenizerTrainRequest, TokenizerTrainResponse
from ..core.config.settings import Settings
from ..core.services.queue.rq_adapter import RQEnqueuer


@dataclass(frozen=True)
class TokenizerEnqueueOut:
    tokenizer_id: str
    job_id: str


class TokenizerOrchestrator:
    def __init__(self, *, settings: Settings, redis_client: redis.Redis[str], enqueuer: RQEnqueuer) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer

    def enqueue_training(self, req: TokenizerTrainRequest) -> TokenizerTrainResponse | None:
        # Placeholder: generate an id and return immediately; actual work will be added
        tokenizer_id = f"tok-{abs(hash((req.method, req.vocab_size, req.corpus_path))) % (10**10):010d}"
        # For now, mark as created; enqueue actual training when backend is implemented
        self._redis.set(f"tokenizer:{tokenizer_id}:status", "queued")
        return TokenizerTrainResponse(tokenizer_id=tokenizer_id, artifact_path=f"/data/artifacts/tokenizers/{tokenizer_id}")

