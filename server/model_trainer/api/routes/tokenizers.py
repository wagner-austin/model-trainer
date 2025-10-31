from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.services.container import ServiceContainer
from ..schemas.tokenizers import TokenizerTrainRequest, TokenizerTrainResponse

def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()

    @router.post("/train", response_model=TokenizerTrainResponse)
    def start_tokenizer_training(req: TokenizerTrainRequest) -> TokenizerTrainResponse:
        orchestrator = container.tokenizer_orchestrator
        out = orchestrator.enqueue_training(req)
        if out is None:
            raise HTTPException(status_code=500, detail="tokenizer training enqueue failed")
        return out

    return router
