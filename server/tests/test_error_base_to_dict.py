"""Test AppError.to_dict method for complete coverage."""

from __future__ import annotations

from model_trainer.core.errors.base import AppError, ErrorCode


def test_app_error_to_dict() -> None:
    """Test AppError.to_dict serializes error code and message correctly."""
    err = AppError(ErrorCode.INTERNAL, "Something went wrong")
    result = err.to_dict()
    assert result == {"error": "INTERNAL", "message": "Something went wrong"}
    assert isinstance(result, dict)
    assert isinstance(result["error"], str)
    assert isinstance(result["message"], str)
