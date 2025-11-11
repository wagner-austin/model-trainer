from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.health import ReadyzResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_readyz_with_redis_disabled(monkeypatch: MonkeyPatch) -> None:
    # Disable Redis via env so readyz short-circuits to ready
    monkeypatch.setenv("REDIS__ENABLED", "false")
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    # Sanity: the app settings should reflect disabled redis
    assert container.settings.redis.enabled is False
    client = TestClient(app)

    r = client.get("/readyz")
    assert r.status_code == 200
    body = ReadyzResponse.model_validate_json(r.text)
    assert body.status == "ready"
