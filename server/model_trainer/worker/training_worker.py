from __future__ import annotations

import logging
import os
from typing import Literal

import redis
from pydantic import BaseModel
from redis.exceptions import RedisError

from ..core.config.settings import Settings
from ..core.contracts.compute import LocalCPUProvider
from ..core.contracts.model import ModelTrainConfig
from ..core.contracts.queue import EvalJobPayload, TrainJobPayload, TrainRequestPayload
from ..core.infra.paths import model_dir, model_logs_path
from ..core.logging.service import LoggingService
from ..core.services.container import ServiceContainer
from ..core.services.tokenizer.bpe_backend import BPEBackend
from ..core.services.tokenizer.spm_backend import SentencePieceBackend
from ..events.trainer import encode_event

HEARTBEAT_KEY_PREFIX = "runs:hb:"
STATUS_KEY_PREFIX = "runs:status:"
EVAL_KEY_PREFIX = "runs:eval:"
MSG_KEY_PREFIX = "runs:msg:"
EVENTS_CHANNEL = "trainer:events"


def _redis_client() -> redis.Redis[str]:
    # Worker environment must provide REDIS_URL
    import os

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def _publish(r: redis.Redis[str], message: str) -> None:
    try:
        r.publish(EVENTS_CHANNEL, message)
    except (RedisError, OSError, ValueError) as e:  # pragma: no cover - best effort publish
        logging.getLogger(__name__).warning("event publish failed: %s", e)


def _setup_env(settings: Settings) -> int:
    threads_cfg = settings.app.threads
    threads = threads_cfg if threads_cfg and threads_cfg > 0 else max(1, int(os.cpu_count() or 1))
    env = LocalCPUProvider(threads_count=threads).env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    try:
        import torch as _torch

        _torch.set_num_threads(threads)
        _torch.set_num_interop_threads(max(1, threads // 2))
    except (ImportError, AttributeError, RuntimeError, ValueError) as _e:  # pragma: no cover
        logging.getLogger(__name__).warning("Failed to set torch threading: %s", _e)
    return threads


def _build_cfg(req: TrainRequestPayload) -> ModelTrainConfig:
    return ModelTrainConfig(
        model_family=req["model_family"],
        model_size=req["model_size"],
        max_seq_len=req["max_seq_len"],
        num_epochs=req["num_epochs"],
        batch_size=req["batch_size"],
        learning_rate=req["learning_rate"],
        tokenizer_id=req["tokenizer_id"],
        corpus_path=req["corpus_path"],
    )


def _emit_started_event(
    r: redis.Redis[str],
    run_id: str,
    user_id: int,
    cfg: ModelTrainConfig,
    threads: int,
) -> None:
    _publish(
        r,
        encode_event(
            {
                "type": "trainer.train.started.v1",
                "request_id": run_id,
                "run_id": run_id,
                "user_id": user_id,
                "model_family": cfg.model_family,
                "model_size": cfg.model_size,
                "total_epochs": cfg.num_epochs,
                "queue": "training",
                "cpu_cores": int(os.cpu_count() or 1),
                "optimal_threads": threads,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
            }
        ),
    )


def _emit_progress_event(
    r: redis.Redis[str],
    run_id: str,
    user_id: int,
    epoch: int,
    total_epochs: int,
    step: int,
    loss: float,
) -> None:
    _publish(
        r,
        encode_event(
            {
                "type": "trainer.train.progress.v1",
                "request_id": run_id,
                "run_id": run_id,
                "user_id": user_id,
                "epoch": int(epoch),
                "total_epochs": int(total_epochs),
                "step": int(step),
                "loss": float(loss),
            }
        ),
    )


def _emit_completed_event(
    r: redis.Redis[str],
    run_id: str,
    user_id: int,
    loss: float,
    perplexity: float,
    artifact_path: str,
) -> None:
    _publish(
        r,
        encode_event(
            {
                "type": "trainer.train.completed.v1",
                "request_id": run_id,
                "run_id": run_id,
                "user_id": user_id,
                "loss": float(loss),
                "perplexity": float(perplexity),
                "artifact_path": artifact_path,
            }
        ),
    )


def _emit_failed_event(
    r: redis.Redis[str],
    run_id: str,
    user_id: int,
    message: str,
    status: Literal["failed", "canceled"],
) -> None:
    _publish(
        r,
        encode_event(
            {
                "type": "trainer.train.failed.v1",
                "request_id": run_id,
                "run_id": run_id,
                "user_id": user_id,
                "error_kind": "system",
                "message": message,
                "status": status,
            }
        ),
    )


def process_train_job(payload: TrainJobPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    run_id = payload["run_id"]
    user_id = int(payload["user_id"])
    r.set(f"{STATUS_KEY_PREFIX}{run_id}", "running")
    settings = Settings()
    threads = _setup_env(settings)

    req = payload["request"]
    cfg = _build_cfg(req)

    def _hb(ts: float) -> None:
        r.set(f"{HEARTBEAT_KEY_PREFIX}{run_id}", str(ts))

    def _cancelled() -> bool:
        val = r.get(f"runs:{run_id}:cancelled")
        return bool(val == "1")

    try:
        # Per-run structured log (attach before training starts)
        logsvc = LoggingService.create()
        run_log_path = str(model_logs_path(settings, run_id))
        per_run_logger = logsvc.attach_run_file(
            path=run_log_path, category="training", service="worker", run_id=run_id
        )
        # Initial heartbeat and config logging
        import time as _time

        _hb(_time.time())
        per_run_logger.info(
            "Training started",
            extra={
                "event": "train_started",
                "run_id": run_id,
                "model_family": cfg.model_family,
                "model_size": cfg.model_size,
                "max_seq_len": cfg.max_seq_len,
                "num_epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "tokenizer_id": cfg.tokenizer_id,
                "corpus_path": cfg.corpus_path,
                "steps": 0,
            },
        )
        # Publish started event
        _emit_started_event(r, run_id, user_id, cfg, threads)

        def _progress(step: int, epoch: int, loss: float) -> None:
            per_run_logger.info(
                "Training progress",
                extra={"event": "train_progress", "run_id": run_id, "steps": step, "loss": loss},
            )
            _emit_progress_event(
                r,
                run_id,
                user_id,
                int(epoch),
                cfg.num_epochs,
                int(step),
                float(loss),
            )

        # Use model registry for backend selection
        container = ServiceContainer.from_settings(settings)
        backend = container.model_registry.get(cfg.model_family)
        # Prepare model using a tokenizer handle from artifacts
        tok_dir = os.path.join(settings.app.artifacts_root, "tokenizers", cfg.tokenizer_id)
        tok_json = os.path.join(tok_dir, "tokenizer.json")
        tok_spm = os.path.join(tok_dir, "tokenizer.model")
        if os.path.exists(tok_json):
            tok_handle = BPEBackend().load(tok_json)
        elif os.path.exists(tok_spm):
            tok_handle = SentencePieceBackend().load(tok_spm)
        else:
            raise FileNotFoundError(
                f"Tokenizer artifact not found: expected {tok_json} or {tok_spm}"
            )
        prepared = backend.prepare(cfg, settings, tokenizer=tok_handle)
        result = backend.train(
            cfg,
            settings,
            run_id=run_id,
            heartbeat=_hb,
            cancelled=_cancelled,
            prepared=prepared,
            progress=_progress,
        )
        if result.cancelled:
            # Mark failed on cancellation and do not persist weights
            r.set(f"{STATUS_KEY_PREFIX}{run_id}", "failed")
            r.set(f"{MSG_KEY_PREFIX}{run_id}", "Training cancelled")
            per_run_logger.info(
                "Training cancelled",
                extra={
                    "event": "train_cancelled",
                    "run_id": run_id,
                    "loss": result.loss,
                    "perplexity": result.perplexity,
                    "steps": result.steps,
                },
            )
            _emit_failed_event(r, run_id, user_id, "Training cancelled", "canceled")
        else:
            r.set(f"{STATUS_KEY_PREFIX}{run_id}", "completed")
            r.set(f"{MSG_KEY_PREFIX}{run_id}", "Training completed")
            # Save weights via backend lifecycle
            out_dir = str(model_dir(settings, run_id))
            _ = backend.save(prepared, out_dir)
            per_run_logger.info(
                "Training completed",
                extra={
                    "event": "train_completed",
                    "run_id": run_id,
                    "loss": result.loss,
                    "perplexity": result.perplexity,
                    "steps": result.steps,
                },
            )
            _emit_completed_event(
                r,
                run_id,
                user_id,
                float(result.loss),
                float(result.perplexity),
                out_dir,
            )
        # Release handler for long-running worker process
        logsvc.close_run_file(path=run_log_path)
    except Exception as e:
        r.set(f"{STATUS_KEY_PREFIX}{run_id}", "failed")
        try:
            r.set(f"{MSG_KEY_PREFIX}{run_id}", str(e))
        except (RedisError, OSError, ValueError, TypeError) as pub_exc:
            logging.getLogger(__name__).warning("failed to set status message: %s", pub_exc)
            raise e from pub_exc
        log.exception("Training job failed run_id=%s error=%s", run_id, e)
        try:
            _emit_failed_event(r, run_id, user_id, str(e), "failed")
        except (RedisError, OSError, ValueError) as pub_exc:
            logging.getLogger(__name__).warning("failed to publish failure event: %s", pub_exc)
        raise


class _EvalCacheModel(BaseModel):
    status: str
    split: str
    loss: float | None = None
    ppl: float | None = None
    artifact: str | None = None


def process_eval_job(payload: EvalJobPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    run_id = payload["run_id"]
    split = payload["split"]
    running = _EvalCacheModel(status="running", split=split)
    r.set(f"{EVAL_KEY_PREFIX}{run_id}", running.model_dump_json())
    # Evaluate using saved model and conservative defaults if full cfg unavailable
    settings = Settings()
    # Load training manifest for this run to get tokenizer_id and params
    from pathlib import Path

    artifacts_root = settings.app.artifacts_root
    manifest_path = Path(artifacts_root) / "models" / run_id / "manifest.json"
    if not manifest_path.exists():
        r.set(
            f"{EVAL_KEY_PREFIX}{run_id}",
            _EvalCacheModel(status="failed", split=split).model_dump_json(),
        )
        logging.getLogger(__name__).error("Eval failed: manifest missing for run_id=%s", run_id)
        return

    class _ManifestVersions(BaseModel):
        torch: str
        transformers: str
        tokenizers: str
        datasets: str

        model_config = {"extra": "forbid", "validate_assignment": True}

    class _ManifestSystem(BaseModel):
        cpu_count: int
        platform: str
        platform_release: str
        machine: str

        model_config = {"extra": "forbid", "validate_assignment": True}

    class _TrainingManifestModel(BaseModel):
        run_id: str
        model_family: str
        model_size: str
        epochs: int
        batch_size: int
        max_seq_len: int
        steps: int
        loss: float
        learning_rate: float
        tokenizer_id: str
        corpus_path: str
        optimizer: str
        seed: int
        versions: _ManifestVersions
        system: _ManifestSystem
        git_commit: str | None = None

        model_config = {"extra": "forbid", "validate_assignment": True}

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = _TrainingManifestModel.model_validate_json(manifest_text)
    cfg = ModelTrainConfig(
        model_family=manifest.model_family,
        model_size=manifest.model_size,
        max_seq_len=manifest.max_seq_len,
        num_epochs=manifest.epochs,
        batch_size=manifest.batch_size,
        learning_rate=manifest.learning_rate,
        tokenizer_id=manifest.tokenizer_id,
        corpus_path=manifest.corpus_path,
    )
    try:
        container2 = ServiceContainer.from_settings(settings)
        backend = container2.model_registry.get("gpt2")
        res = backend.evaluate(run_id=run_id, cfg=cfg, settings=settings)
        # Persist path to metrics.json as artifact pointer
        from ..core.infra.paths import model_eval_dir as _model_eval_dir

        artifact_path = str(_model_eval_dir(settings, run_id) / "metrics.json")
        out = _EvalCacheModel(
            status="completed",
            split=split,
            loss=res.loss,
            ppl=res.perplexity,
            artifact=artifact_path,
        )
    except Exception as e:
        out = _EvalCacheModel(status="failed", split=split, loss=None, ppl=None)
        logging.getLogger(__name__).exception("Eval failed run_id=%s error=%s", run_id, e)
        r.set(f"{EVAL_KEY_PREFIX}{run_id}", out.model_dump_json())
        raise
    else:
        r.set(f"{EVAL_KEY_PREFIX}{run_id}", out.model_dump_json())
        log.info("Eval job completed run_id=%s split=%s", run_id, split)
