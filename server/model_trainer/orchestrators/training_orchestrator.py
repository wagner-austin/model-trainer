from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import redis

from ..api.schemas.runs import (
    EvaluateRequest,
    EvaluateResponse,
    RunStatusResponse,
    TrainRequest,
)
from ..core.config.settings import Settings
from ..core.contracts.queue import EvalJobPayload, TrainJobPayload, TrainRequestPayload
from ..core.logging.service import LoggingService
from ..core.services.queue.rq_adapter import RQEnqueuer
from ..infra.persistence.models import EvalCache
from ..infra.storage.run_store import RunStore

HEARTBEAT_KEY_PREFIX: Final[str] = "runs:hb:"
STATUS_KEY_PREFIX: Final[str] = "runs:status:"
EVAL_KEY_PREFIX: Final[str] = "runs:eval:"


@dataclass
class EnqueueOut:
    run_id: str
    job_id: str


class TrainingOrchestrator:
    def __init__(
        self: TrainingOrchestrator,
        *,
        settings: Settings,
        redis_client: redis.Redis[str],
        enqueuer: RQEnqueuer,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer
        self._store = RunStore(settings.app.runs_root, settings.app.artifacts_root)
        self._logger = LoggingService.create().adapter(category="orchestrator", service="training")

    def enqueue_training(self: TrainingOrchestrator, req: TrainRequest) -> EnqueueOut:
        run_id = self._store.create_run(req.model_family, req.model_size)
        request_payload: TrainRequestPayload = {
            "model_family": req.model_family,
            "model_size": req.model_size,
            "max_seq_len": req.max_seq_len,
            "num_epochs": req.num_epochs,
            "batch_size": req.batch_size,
            "learning_rate": req.learning_rate,
            "corpus_path": req.corpus_path,
            "tokenizer_id": req.tokenizer_id,
        }
        payload: TrainJobPayload = {"run_id": run_id, "request": request_payload}
        # Per-run log file
        run_log_path = os.path.join(
            self._settings.app.artifacts_root, "models", run_id, "logs.jsonl"
        )
        per_run_logger = LoggingService.create().attach_run_file(
            path=run_log_path, category="training", service="orchestrator", run_id=run_id
        )

        job_id = self._enq.enqueue_train(payload)
        self._redis.set(f"{STATUS_KEY_PREFIX}{run_id}", "queued")
        per_run_logger.info("training enqueued", extra={"event": "enqueued", "run_id": run_id})
        return EnqueueOut(run_id=run_id, job_id=job_id)

    def get_status(self: TrainingOrchestrator, run_id: str) -> RunStatusResponse | None:
        status_v = self._redis.get(f"{STATUS_KEY_PREFIX}{run_id}")
        if status_v is None:
            return None
        hb_raw = self._redis.get(f"{HEARTBEAT_KEY_PREFIX}{run_id}")
        hb = float(hb_raw) if hb_raw is not None else None
        return RunStatusResponse(run_id=run_id, status=status_v, last_heartbeat_ts=hb)

    def enqueue_evaluation(
        self: TrainingOrchestrator, run_id: str, req: EvaluateRequest
    ) -> EvaluateResponse:
        key = f"{STATUS_KEY_PREFIX}{run_id}"
        if self._redis.get(key) is None:
            return EvaluateResponse(
                run_id=run_id, split=req.split, status="failed", loss=None, perplexity=None
            )
        payload: EvalJobPayload = {
            "run_id": run_id,
            "split": req.split,
            "path_override": req.path_override,
        }
        run_log_path = os.path.join(
            self._settings.app.artifacts_root, "models", run_id, "logs.jsonl"
        )
        per_run_logger = LoggingService.create().attach_run_file(
            path=run_log_path, category="training", service="orchestrator", run_id=run_id
        )

        _ = self._enq.enqueue_eval(payload)
        cache = EvalCache(status="queued", split=req.split, loss=None, ppl=None, artifact=None)
        self._redis.set(f"{EVAL_KEY_PREFIX}{run_id}", cache.model_dump_json())
        per_run_logger.info(
            "eval enqueued", extra={"event": "eval_enqueued", "run_id": run_id, "split": req.split}
        )
        return EvaluateResponse(
            run_id=run_id, split=req.split, status="queued", loss=None, perplexity=None
        )

    def get_evaluation(self: TrainingOrchestrator, run_id: str) -> EvaluateResponse | None:
        raw = self._redis.get(f"{EVAL_KEY_PREFIX}{run_id}")
        if raw is None:
            return None
        cache = EvalCache.model_validate_json(raw)
        return EvaluateResponse(
            run_id=run_id,
            split=cache.split,
            status=cache.status,
            loss=cache.loss,
            perplexity=cache.ppl,
            artifact_path=cache.artifact,
        )
