from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.runs import EvaluateResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer
from model_trainer.infra.persistence.models import EvalCache


def test_get_eval_returns_artifact_pointer() -> None:
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    fake = fakeredis.FakeRedis(decode_responses=True)
    container.redis = fake
    container.training_orchestrator._redis = fake

    run_id = "run-eval"
    # Pre-populate evaluation cache to simulate a completed eval
    cache = EvalCache(
        status="completed",
        split="validation",
        loss=0.5,
        ppl=1.5,
        artifact="/data/artifacts/models/run-eval/eval/metrics.json",
    )
    fake.set(f"runs:eval:{run_id}", cache.model_dump_json())

    client = TestClient(app)
    r = client.get(f"/runs/{run_id}/eval")
    assert r.status_code == 200
    body = EvaluateResponse.model_validate_json(r.text)
    assert body.status == "completed"
    assert body.artifact_path is not None
    assert body.artifact_path.endswith("/models/run-eval/eval/metrics.json")
