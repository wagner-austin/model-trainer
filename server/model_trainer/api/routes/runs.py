from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.services.container import ServiceContainer
from ..schemas.runs import (
    TrainRequest,
    TrainResponse,
    RunStatusResponse,
    EvaluateRequest,
    EvaluateResponse,
)

def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()

    @router.post("/train", response_model=TrainResponse)
    def start_training(req: TrainRequest) -> TrainResponse:
        orchestrator = container.training_orchestrator
        out = orchestrator.enqueue_training(req)
        return TrainResponse(run_id=out.run_id, job_id=out.job_id)

    @router.get("/{run_id}", response_model=RunStatusResponse)
    def run_status(run_id: str) -> RunStatusResponse:
        orchestrator = container.training_orchestrator
        status_out = orchestrator.get_status(run_id)
        if status_out is None:
            raise HTTPException(status_code=404, detail="run not found")
        return status_out

    @router.post("/{run_id}/evaluate", response_model=EvaluateResponse)
    def run_evaluate(run_id: str, req: EvaluateRequest) -> EvaluateResponse:
        orchestrator = container.training_orchestrator
        return orchestrator.enqueue_evaluation(run_id, req)

    @router.get("/{run_id}/eval", response_model=EvaluateResponse)
    def run_eval_result(run_id: str) -> EvaluateResponse:
        orchestrator = container.training_orchestrator
        result = orchestrator.get_evaluation(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="eval not found")
        return result

    return router
