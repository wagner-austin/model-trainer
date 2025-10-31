from __future__ import annotations

import fakeredis
import model_trainer.api.routes.health as health
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.health import ReadyzResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def _app_with_fake_redis() -> TestClient:
    app = create_app(Settings())
    fake = fakeredis.FakeRedis(decode_responses=True)
    # replace container redis with fake
    container: ServiceContainer = app.state.container
    container.redis = fake
    return TestClient(app)


def test_readyz_degraded_without_worker(monkeypatch: MonkeyPatch) -> None:
    client = _app_with_fake_redis()

    # No workers registered
    class _W:
        @staticmethod
        def all(conn: object) -> list[object]:
            return []

    monkeypatch.setattr(health, "_Worker", _W)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = ReadyzResponse.model_validate_json(resp.text)
    assert body.status == "degraded"
    assert body.reason in ("no-worker", "redis no-pong", "redis error")


def test_readyz_ready_with_worker(monkeypatch: MonkeyPatch) -> None:
    client = _app_with_fake_redis()

    # One worker present
    class _W2:
        @staticmethod
        def all(conn: object) -> list[object]:
            return [object()]

    monkeypatch.setattr(health, "_Worker", _W2)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = ReadyzResponse.model_validate_json(resp.text)
    assert body.status == "ready"
