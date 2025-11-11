from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ...api.request_context import request_id_var
from .base import AppError, ErrorCode


def install_exception_handlers(app: FastAPI) -> None:
    async def _app_error_handler(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        rid = request_id_var.get()
        if isinstance(exc, AppError):
            status_map = {
                ErrorCode.DATA_NOT_FOUND: 404,
                ErrorCode.CONFIG_INVALID: 400,
                ErrorCode.UNAUTHORIZED: 401,
                ErrorCode.TOKENIZER_TRAIN_FAILED: 500,
                ErrorCode.MODEL_TRAIN_FAILED: 500,
                ErrorCode.INTERNAL: 500,
            }
            payload: dict[str, str] = {
                "error": exc.code.value,
                "code": exc.code.value,
                "message": exc.message,
                "request_id": rid,
            }
            return JSONResponse(content=payload, status_code=status_map.get(exc.code, 500))
        payload2: dict[str, str] = {
            "error": ErrorCode.INTERNAL.value,
            "code": ErrorCode.INTERNAL.value,
            "message": str(exc),
            "request_id": rid,
        }
        return JSONResponse(content=payload2, status_code=500)

    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        rid = request_id_var.get()
        payload: dict[str, str] = {
            "code": ErrorCode.INTERNAL.value,
            "message": str(exc),
            "request_id": rid,
        }
        return JSONResponse(content=payload, status_code=500)

    # Explicit registration instead of decorators
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(Exception, _unhandled)
