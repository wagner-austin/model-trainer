from __future__ import annotations

import logging

from fastapi import FastAPI

from ..core.config.settings import Settings
from ..core.errors.handlers import install_exception_handlers
from ..core.logging.setup import setup_logging
from ..core.services.container import ServiceContainer
from .middleware import RequestIdMiddleware, api_key_dependency
from .routes import artifacts, health, runs, tokenizers


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    setup_logging(cfg.logging.level)
    app = FastAPI(title="Model Trainer API", version="0.1.0")

    container = ServiceContainer.from_settings(cfg)
    # Expose container for testability and tooling
    app.state.container = container

    # Middleware: request correlation and strict API key enforcement
    app.add_middleware(RequestIdMiddleware)
    # FastAPI dependency for API key (required); attach to routers where appropriate
    app.state.api_key_dep = api_key_dependency(cfg)

    # Routers (container captured in closures)
    app.include_router(health.build_router(container), prefix="")
    app.include_router(runs.build_router(container), prefix="/runs", tags=["runs"])
    app.include_router(
        tokenizers.build_router(container), prefix="/tokenizers", tags=["tokenizers"]
    )
    app.include_router(artifacts.build_router(container), tags=["artifacts"])

    # Errors
    install_exception_handlers(app)

    logging.getLogger(__name__).info("API application initialized")
    return app
