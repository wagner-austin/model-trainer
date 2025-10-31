from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    TOKENIZER_TRAIN_FAILED = "TOKENIZER_TRAIN_FAILED"
    MODEL_TRAIN_FAILED = "MODEL_TRAIN_FAILED"
    CONFIG_INVALID = "CONFIG_INVALID"
    INTERNAL = "INTERNAL"


class AppError(Exception):
    def __init__(self: AppError, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self: AppError) -> dict[str, str]:
        return {"error": self.code.value, "message": self.message}
