from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .base import AppError, ErrorCode


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:  # pragma: no cover
        status_map = {
            ErrorCode.DATA_NOT_FOUND: 404,
            ErrorCode.CONFIG_INVALID: 400,
            ErrorCode.TOKENIZER_TRAIN_FAILED: 500,
            ErrorCode.MODEL_TRAIN_FAILED: 500,
            ErrorCode.INTERNAL: 500,
        }
        return JSONResponse(content=exc.to_dict(), status_code=status_map.get(exc.code, 500))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        return JSONResponse(
            content={"error": ErrorCode.INTERNAL.value, "message": str(exc)}, status_code=500
        )

