from __future__ import annotations

import redis as _redis
from fastapi import APIRouter, Response, status
from rq import Worker as _Worker

from ...core.config.settings import Settings
from ...core.services.container import ServiceContainer
from ..schemas.health import HealthzResponse, ReadyzResponse


def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()

    def healthz() -> HealthzResponse:
        container.logging.adapter(category="api", service="health").info(
            "healthz", extra={"event": "healthz"}
        )
        return HealthzResponse(status="ok")

    def readyz(response: Response) -> ReadyzResponse:
        from redis.exceptions import RedisError

        settings: Settings = container.settings
        try:
            if settings.redis.enabled:
                client: _redis.Redis[str] = container.redis
                pong = client.ping()
                if not pong:
                    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    container.logging.adapter(category="api", service="health").info(
                        "readyz degraded", extra={"event": "readyz", "reason": "redis no-pong"}
                    )
                    return ReadyzResponse(status="degraded", reason="redis no-pong")
                # Worker presence check: at least one RQ worker registered
                workers = _Worker.all(client)
                if len(workers) == 0:
                    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    container.logging.adapter(category="api", service="health").info(
                        "readyz degraded", extra={"event": "readyz", "reason": "no-worker"}
                    )
                    return ReadyzResponse(status="degraded", reason="no-worker")
        except RedisError:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            container.logging.adapter(category="api", service="health").info(
                "readyz degraded", extra={"event": "readyz", "reason": "redis error"}
            )
            return ReadyzResponse(status="degraded", reason="redis error")
        container.logging.adapter(category="api", service="health").info(
            "readyz", extra={"event": "readyz", "status": "ready"}
        )
        return ReadyzResponse(status="ready")

    router.add_api_route("/healthz", healthz, methods=["GET"], response_model=HealthzResponse)
    router.add_api_route("/readyz", readyz, methods=["GET"], response_model=ReadyzResponse)
    return router
