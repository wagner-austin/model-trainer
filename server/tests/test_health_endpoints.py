from __future__ import annotations

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.health import ReadyzResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def _client_and_container() -> tuple[TestClient, ServiceContainer]:
    app = create_app(Settings())
    fake = fakeredis.FakeRedis(decode_responses=True)
    # replace container redis with fake
    container: ServiceContainer = app.state.container
    container.redis = fake
    return TestClient(app), container


def test_readyz_degraded_without_worker(monkeypatch: MonkeyPatch) -> None:
    client, _ = _client_and_container()
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = ReadyzResponse.model_validate_json(resp.text)
    assert body.status == "degraded"
    assert body.reason in ("no-worker", "redis no-pong", "redis error")


def test_readyz_ready_with_worker(monkeypatch: MonkeyPatch) -> None:
    client, container = _client_and_container()
    # Simulate a worker registered in RQ registry set
    container.redis.sadd("rq:workers", "worker:1")
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = ReadyzResponse.model_validate_json(resp.text)
    assert body.status == "ready"
