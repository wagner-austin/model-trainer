from __future__ import annotations

from pathlib import Path

import fakeredis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.routes import runs as runs_routes
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.errors.handlers import install_exception_handlers
from model_trainer.core.logging.service import LoggingService
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from pytest import MonkeyPatch


def _mk_app(tmp: Path, monkeypatch: MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp / "artifacts"))
    monkeypatch.setenv("APP__RUNS_ROOT", str(tmp / "runs"))
    s = Settings()

    # Fake Redis everywhere
    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fakeredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr("redis.from_url", _fake_from_url)

    # Short-circuit RQ enqueues
    def _enq_train(self: RQEnqueuer, payload: dict[str, object]) -> str:
        return "job-train-1"

    def _enq_eval(self: RQEnqueuer, payload: dict[str, object]) -> str:
        return "job-eval-1"

    monkeypatch.setattr(RQEnqueuer, "enqueue_train", _enq_train)
    monkeypatch.setattr(RQEnqueuer, "enqueue_eval", _enq_eval)
    app = create_app(s)
    return TestClient(app)


def test_runs_logs_not_found(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _mk_app(tmp_path, monkeypatch)
    res = client.get("/runs/unknown/logs")
    assert res.status_code == 404


def test_runs_logs_read_failure(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _mk_app(tmp_path, monkeypatch)
    # Create a directory at the expected log path to trigger OSError on open
    s = Settings()
    log_dir = Path(s.app.artifacts_root) / "models" / "r1"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "logs.jsonl").mkdir(parents=True, exist_ok=True)
    res = client.get("/runs/r1/logs")
    assert res.status_code == 500


def test_runs_logs_tail_content(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _mk_app(tmp_path, monkeypatch)
    s = Settings()
    run_dir = Path(s.app.artifacts_root) / "models" / "r2"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs.jsonl"
    log_path.write_text("a\n" * 10 + "b\n" + "c\n", encoding="utf-8")
    res = client.get("/runs/r2/logs?tail=2")
    assert res.status_code == 200
    assert res.text.strip().splitlines() == ["b", "c"]


def test_runs_logs_stream_not_found(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _mk_app(tmp_path, monkeypatch)
    res = client.get("/runs/nope/logs/stream")
    assert res.status_code == 404


def test_runs_logs_stream_follow_false(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _mk_app(tmp_path, monkeypatch)
    s = Settings()
    run_dir = Path(s.app.artifacts_root) / "models" / "r3"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs.jsonl").write_text("x\n" * 3, encoding="utf-8")
    with client.stream("GET", "/runs/r3/logs/stream?tail=2&follow=false") as resp:
        assert resp.status_code == 200
        body = b"".join(list(resp.iter_bytes()))
    # Expect two SSE data lines
    lines = [ln for ln in body.split(b"\n\n") if ln]
    assert len(lines) == 2
    assert lines[0].startswith(b"data: ")


def test_runs_eval_result_not_found(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client = _mk_app(tmp_path, monkeypatch)
    res = client.get("/runs/unknown/eval")
    # Exception handler maps to 404
    assert res.status_code == 404


def test_runs_train_unsupported_backend_maps_400(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Build an app with an empty model registry to force orchestrator error
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("APP__RUNS_ROOT", str(tmp_path / "runs"))
    s = Settings()

    # Fake redis
    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fakeredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr("redis.from_url", _fake_from_url)

    # Enqueue methods
    def _enq_train(self: RQEnqueuer, payload: TrainJobPayload) -> str:
        return "job-x"

    monkeypatch.setattr(RQEnqueuer, "enqueue_train", _enq_train)
    enq = RQEnqueuer("redis://ignored", RQSettings("q", 1, 1, 1, 0, []))
    r = fakeredis.FakeRedis(decode_responses=True)
    model_reg = ModelRegistry(backends={})
    tokenizer_reg = TokenizerRegistry(backends={})
    training = TrainingOrchestrator(
        settings=s,
        redis_client=r,
        enqueuer=enq,
        model_registry=model_reg,
    )
    tokenizer = TokenizerOrchestrator(settings=s, redis_client=r, enqueuer=enq)
    container = ServiceContainer(
        settings=s,
        redis=r,
        rq_enqueuer=enq,
        training_orchestrator=training,
        tokenizer_orchestrator=tokenizer,
        model_registry=model_reg,
        tokenizer_registry=tokenizer_reg,
        logging=LoggingService.create(),
        dataset_builder=LocalTextDatasetBuilder(),
    )
    app = FastAPI()
    app.include_router(runs_routes.build_router(container), prefix="/runs")
    install_exception_handlers(app)
    client = TestClient(app)
    payload = {
        "model_family": "llama",
        "model_size": "s",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_path": str(tmp_path),
        "tokenizer_id": "tok",
    }
    res = client.post("/runs/train", json=payload)
    assert res.status_code == 400
