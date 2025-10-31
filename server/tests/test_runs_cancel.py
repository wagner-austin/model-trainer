from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_cancel_endpoint_sets_flag() -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake

    client = TestClient(app)
    run_id = "run-x"
    r = client.post(f"/runs/{run_id}/cancel")
    assert r.status_code == 200
    assert fake.get(f"runs:{run_id}:cancelled") == "1"
