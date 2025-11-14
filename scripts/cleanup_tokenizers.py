from __future__ import annotations

import logging

from model_trainer.core.config.settings import Settings
from model_trainer.core.logging.setup import setup_logging
from model_trainer.core.services.tokenizer.tokenizer_cleanup import (
    TokenizerCleanupError,
    TokenizerCleanupService,
)


def main() -> int:
    settings = Settings()
    setup_logging(settings.logging.level)
    logger = logging.getLogger(__name__)
    service = TokenizerCleanupService(settings=settings)
    logger.info(
        "Tokenizer cleanup CLI starting",
        extra={"event": "tokenizer_cleanup_cli_started"},
    )
    try:
        result = service.clean()
    except TokenizerCleanupError as exc:
        logger.error(
            "Tokenizer cleanup CLI failed",
            extra={
                "event": "tokenizer_cleanup_cli_failed",
                "error": str(exc),
            },
        )
        return 1
    logger.info(
        "Tokenizer cleanup CLI completed",
        extra={
            "event": "tokenizer_cleanup_cli_completed",
            "deleted_tokenizers": result.deleted_tokenizers,
            "bytes_freed": result.bytes_freed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

