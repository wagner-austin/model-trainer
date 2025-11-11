from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_trainer.api.middleware import RequestIdMiddleware
from model_trainer.core.errors.base import AppError, ErrorCode
from model_trainer.core.errors.handlers import install_exception_handlers
from pydantic import BaseModel


def test_error_handler_includes_request_id() -> None:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)

    def boom() -> None:
        raise AppError(ErrorCode.CONFIG_INVALID, "nope")

    app.add_api_route("/boom", boom, methods=["GET"])  # avoid decorator type inference pitfalls

    client = TestClient(app)
    r = client.get("/boom", headers={"X-Request-ID": "abc"})
    assert r.status_code == 400

    class _Err(BaseModel):
        code: str
        message: str
        request_id: str

    body = _Err.model_validate_json(r.text)
    assert body.code == "CONFIG_INVALID" and body.request_id == "abc"
