from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer
from pydantic import BaseModel


def test_run_status_not_found_maps_to_app_error() -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake
    client = TestClient(app)

    r = client.get("/runs/nonexistent")
    assert r.status_code == 404

    class _Err(BaseModel):
        error: str
        message: str

    body = _Err.model_validate_json(r.text)
    assert body.error == "DATA_NOT_FOUND"
    assert "not found" in body.message.lower()
