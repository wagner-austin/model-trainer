from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.runs import EvaluateRequest, EvaluateResponse
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_evaluate_and_eval_result_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    app = create_app(Settings())
    cont: ServiceContainer = app.state.container

    # Stub orchestrator methods
    def _enq(run_id: str, req: EvaluateRequest) -> EvaluateResponse:
        return EvaluateResponse(
            run_id=run_id, split=req.split, status="queued", loss=None, perplexity=None
        )

    def _get(run_id: str) -> EvaluateResponse:
        return EvaluateResponse(
            run_id=run_id, split="validation", status="completed", loss=1.0, perplexity=2.0
        )

    monkeypatch.setattr(cont.training_orchestrator, "enqueue_evaluation", _enq)
    monkeypatch.setattr(cont.training_orchestrator, "get_evaluation", _get)

    client = TestClient(app)
    run_id = "r-eval"
    payload: dict[str, object] = {"split": "validation", "path_override": None}
    r1 = client.post(f"/runs/{run_id}/evaluate", json=payload)
    assert r1.status_code == 200
    enq = EvaluateResponse.model_validate_json(r1.text)
    assert enq.status == "queued"
    r2 = client.get(f"/runs/{run_id}/eval")
    assert r2.status_code == 200
    got = EvaluateResponse.model_validate_json(r2.text)
    assert got.status == "completed"
