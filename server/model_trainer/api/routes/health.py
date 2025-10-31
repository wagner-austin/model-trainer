from __future__ import annotations

from fastapi import APIRouter, Response, status

from ...core.config.settings import Settings
from ...core.services.container import ServiceContainer

def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    def readyz(response: Response) -> dict[str, str]:
        # Try to touch Redis to ensure dependencies are alive
        from redis.exceptions import RedisError

        settings: Settings = container.settings
        try:
            if settings.redis.enabled:
                client = container.redis
                pong = client.ping()
                if not pong:
                    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                    return {"status": "degraded", "reason": "redis no-pong"}
        except RedisError:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "degraded", "reason": "redis error"}
        return {"status": "ready"}

    return router
