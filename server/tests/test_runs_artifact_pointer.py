from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.pointers import ArtifactPointer
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_artifact_pointer_404_then_200() -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake

    client = TestClient(app)
    run_id = "run-x"

    r1 = client.get(f"/runs/{run_id}/artifact")
    assert r1.status_code == 404

    fake.set(f"runs:artifact:{run_id}:file_id", "deadbeef")
    r2 = client.get(f"/runs/{run_id}/artifact")
    assert r2.status_code == 200
    body = ArtifactPointer.model_validate_json(r2.text)
    assert body.storage == "data-bank" and body.file_id == "deadbeef"
