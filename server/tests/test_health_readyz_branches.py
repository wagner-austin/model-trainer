from __future__ import annotations

import fakeredis
import redis
from fastapi import FastAPI
from model_trainer.api.routes import health
from model_trainer.core.config.settings import Settings
from model_trainer.core.logging.service import LoggingService
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from pydantic import BaseModel
from pytest import MonkeyPatch
from starlette.testclient import TestClient


def _make_container(settings: Settings, r: fakeredis.FakeRedis) -> ServiceContainer:
    enq = RQEnqueuer("redis://ignored", RQSettings("q", 1, 1, 1, 0, []))
    model_reg = ModelRegistry(backends={})
    tok_reg = TokenizerRegistry(backends={})
    training = TrainingOrchestrator(
        settings=settings,
        redis_client=r,
        enqueuer=enq,
        model_registry=model_reg,
    )
    tokenizer = TokenizerOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
    return ServiceContainer(
        settings=settings,
        redis=r,
        rq_enqueuer=enq,
        training_orchestrator=training,
        tokenizer_orchestrator=tokenizer,
        model_registry=model_reg,
        tokenizer_registry=tok_reg,
        logging=LoggingService.create(),
        dataset_builder=LocalTextDatasetBuilder(),
    )


def _build_app(container: ServiceContainer) -> TestClient:
    app = FastAPI()
    app.include_router(health.build_router(container))
    return TestClient(app)


def test_readyz_redis_no_pong(monkeypatch: MonkeyPatch) -> None:
    s = Settings()
    r = fakeredis.FakeRedis(decode_responses=True)

    # Fake .ping() returning False
    def _no_pong() -> bool:
        return False

    monkeypatch.setattr(r, "ping", _no_pong)
    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 503
    assert Readyz.model_validate_json(res.text).status == "degraded"


def test_readyz_no_worker(monkeypatch: MonkeyPatch) -> None:
    s = Settings()
    r = fakeredis.FakeRedis(decode_responses=True)

    def _pong() -> bool:
        return True

    monkeypatch.setattr(r, "ping", _pong)

    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 503
    assert Readyz.model_validate_json(res.text).status == "degraded"


def test_readyz_redis_error(monkeypatch: MonkeyPatch) -> None:
    s = Settings()
    r = fakeredis.FakeRedis(decode_responses=True)

    class _E(redis.exceptions.RedisError):
        pass

    def _raise() -> bool:
        raise _E()

    monkeypatch.setattr(r, "ping", _raise)
    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 503
    assert Readyz.model_validate_json(res.text).status == "degraded"


def test_readyz_ready(monkeypatch: MonkeyPatch) -> None:
    s = Settings()
    r = fakeredis.FakeRedis(decode_responses=True)

    def _pong() -> bool:
        return True

    monkeypatch.setattr(r, "ping", _pong)

    # Simulate presence of one worker in RQ registry set
    r.sadd("rq:workers", "worker:1")
    client = _build_app(_make_container(s, r))
    res = client.get("/readyz")
    assert res.status_code == 200
    assert Readyz.model_validate_json(res.text).status == "ready"


class Readyz(BaseModel):
    status: str
    reason: str | None = None
