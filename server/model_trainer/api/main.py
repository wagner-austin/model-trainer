from __future__ import annotations

import logging
from fastapi import FastAPI

from ..core.config.settings import Settings
from ..core.logging.setup import setup_logging
from ..core.errors.handlers import install_exception_handlers
from ..core.services.container import ServiceContainer
from .routes import health, runs, tokenizers


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    setup_logging(cfg.logging.level)
    app = FastAPI(title="Model Trainer API", version="0.1.0")

    container = ServiceContainer.from_settings(cfg)

    # Routers (container captured in closures)
    app.include_router(health.build_router(container), prefix="")
    app.include_router(runs.build_router(container), prefix="/runs", tags=["runs"])
    app.include_router(tokenizers.build_router(container), prefix="/tokenizers", tags=["tokenizers"])

    # Errors
    install_exception_handlers(app)

    logging.getLogger(__name__).info("API application initialized")
    return app
