from __future__ import annotations

import logging
import os

import redis
from pydantic import BaseModel

from ..core.config.settings import Settings
from ..core.contracts.compute import LocalCPUProvider
from ..core.contracts.queue import EvalJobPayload, TrainJobPayload
from ..core.infra.paths import model_logs_path
from ..core.logging.service import LoggingService
from ..core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from ..core.services.training.gpt2_backend import GPT2TrainConfig, evaluate_gpt2, train_gpt2

HEARTBEAT_KEY_PREFIX = "runs:hb:"
STATUS_KEY_PREFIX = "runs:status:"
EVAL_KEY_PREFIX = "runs:eval:"


def _redis_client() -> redis.Redis[str]:
    # Worker environment must provide REDIS_URL
    import os

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def process_train_job(payload: TrainJobPayload) -> None:
    log = logging.getLogger(__name__)
    r = _redis_client()
    run_id = payload["run_id"]
    r.set(f"{STATUS_KEY_PREFIX}{run_id}", "running")
    settings = Settings()
    # Apply local CPU compute environment
    # Prefer configured threads; fall back to CPU count
    threads_cfg = settings.app.threads
    threads = threads_cfg if threads_cfg and threads_cfg > 0 else max(1, int(os.cpu_count() or 1))
    env = LocalCPUProvider(threads_count=threads).env()
    for k, v in env.items():
        os.environ[k] = v

    req = payload["request"]
    cfg = GPT2TrainConfig(
        model_family="gpt2",
        model_size=req["model_size"],
        max_seq_len=int(req["max_seq_len"]),
        num_epochs=int(req["num_epochs"]),
        batch_size=int(req["batch_size"]),
        learning_rate=float(req["learning_rate"]),
        tokenizer_id=req["tokenizer_id"],
        corpus_path=req["corpus_path"],
    )

    def _hb(ts: float) -> None:
        r.set(f"{HEARTBEAT_KEY_PREFIX}{run_id}", str(ts))

    def _cancelled() -> bool:
        val = r.get(f"runs:{run_id}:cancelled")
        return bool(val == "1")

    try:
        builder = LocalTextDatasetBuilder()
        # Initial heartbeat to indicate job started
        import time as _time

        _hb(_time.time())
        result = train_gpt2(
            cfg,
            settings,
            run_id=run_id,
            redis_hb=_hb,
            cancelled=_cancelled,
            dataset_builder=builder,
        )
        if result.cancelled:
            r.set(f"{STATUS_KEY_PREFIX}{run_id}", "failed")
            log.info("Training cancelled run_id=%s", run_id)
            return
        r.set(f"{STATUS_KEY_PREFIX}{run_id}", "completed")
        # Per-run structured log
        logsvc = LoggingService.create()
        run_log_path = str(model_logs_path(settings, run_id))
        per_run_logger = logsvc.attach_run_file(
            path=run_log_path, category="training", service="worker", run_id=run_id
        )
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
        # Release handler for long-running worker process
        logsvc.close_run_file(path=run_log_path)
    except Exception as e:
        r.set(f"{STATUS_KEY_PREFIX}{run_id}", "failed")
        log.exception("Training job failed run_id=%s error=%s", run_id, e)
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
        epochs: int
        batch_size: int
        max_seq_len: int
        steps: int
        loss: float
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
    tokenizer_id = manifest.tokenizer_id  # no fallback
    max_seq_len = manifest.max_seq_len  # no fallback
    batch_size = manifest.batch_size  # no fallback
    corpus_path = manifest.corpus_path  # no fallback
    cfg = GPT2TrainConfig(
        model_family="gpt2",
        model_size="small",
        max_seq_len=max_seq_len,
        num_epochs=1,
        batch_size=batch_size,
        learning_rate=5e-4,
        tokenizer_id=tokenizer_id,
        corpus_path=corpus_path,
    )
    try:
        builder = LocalTextDatasetBuilder()
        res = evaluate_gpt2(run_id=run_id, cfg=cfg, settings=settings, dataset_builder=builder)
        out = _EvalCacheModel(status="completed", split=split, loss=res.loss, ppl=res.perplexity)
    except Exception as e:
        out = _EvalCacheModel(status="failed", split=split, loss=None, ppl=None)
        logging.getLogger(__name__).exception("Eval failed run_id=%s error=%s", run_id, e)
        r.set(f"{EVAL_KEY_PREFIX}{run_id}", out.model_dump_json())
        raise
    else:
        r.set(f"{EVAL_KEY_PREFIX}{run_id}", out.model_dump_json())
        log.info("Eval job completed run_id=%s split=%s", run_id, split)
