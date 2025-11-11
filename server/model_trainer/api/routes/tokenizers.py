from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Depends as DependsParamType

from ...core.logging.types import LoggingExtra
from ...core.services.container import ServiceContainer
from ..middleware import api_key_dependency
from ..schemas.tokenizers import (
    TokenizerInfoResponse,
    TokenizerTrainRequest,
    TokenizerTrainResponse,
)


def build_router(container: ServiceContainer) -> APIRouter:
    api_dep: DependsParamType = Depends(api_key_dependency(container.settings))
    router = APIRouter(dependencies=[api_dep])

    def start_tokenizer_training(req: TokenizerTrainRequest) -> TokenizerTrainResponse:
        orchestrator = container.tokenizer_orchestrator
        extra: LoggingExtra = {
            "event": "tokenizers_enqueue",
            "method": req.method,
            "vocab_size": req.vocab_size,
        }
        container.logging.adapter(category="api", service="tokenizers").info(
            "tokenizers enqueue", extra=extra
        )
        out = orchestrator.enqueue_training(req)
        if out is None:
            raise HTTPException(status_code=500, detail="tokenizer training enqueue failed")
        return out

    def get_tokenizer(tokenizer_id: str) -> TokenizerInfoResponse:
        r = container.redis
        status = r.get(f"tokenizer:{tokenizer_id}:status") or "unknown"
        stats_json = r.get(f"tokenizer:{tokenizer_id}:stats")
        artifact_path = f"{container.settings.app.artifacts_root}/tokenizers/{tokenizer_id}"
        extra2: LoggingExtra = {"event": "tokenizers_get", "status": status}
        container.logging.adapter(
            category="api", service="tokenizers", run_id=None, tokenizer_id=tokenizer_id
        ).info("tokenizers get", extra=extra2)
        coverage = None
        oov_rate = None
        token_count = None
        char_coverage = None
        if stats_json:
            from pydantic import BaseModel

            class _S(BaseModel):
                coverage: float
                oov_rate: float
                token_count: int
                char_coverage: float

            s = _S.model_validate_json(stats_json)
            coverage = s.coverage
            oov_rate = s.oov_rate
            token_count = s.token_count
            char_coverage = s.char_coverage
        return TokenizerInfoResponse(
            tokenizer_id=tokenizer_id,
            artifact_path=artifact_path,
            status=status,
            coverage=coverage,
            oov_rate=oov_rate,
            token_count=token_count,
            char_coverage=char_coverage,
        )

    router.add_api_route(
        "/train",
        start_tokenizer_training,
        methods=["POST"],
        response_model=TokenizerTrainResponse,
    )
    router.add_api_route(
        "/{tokenizer_id}",
        get_tokenizer,
        methods=["GET"],
        response_model=TokenizerInfoResponse,
    )
    return router
