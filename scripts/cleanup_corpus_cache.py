from __future__ import annotations

import logging

from model_trainer.core.config.settings import Settings
from model_trainer.core.logging.setup import setup_logging
from model_trainer.core.services.data.corpus_cache_cleanup import (
    CorpusCacheCleanupError,
    CorpusCacheCleanupService,
)


def main() -> int:
    settings = Settings()
    setup_logging(settings.logging.level)
    logger = logging.getLogger(__name__)
    service = CorpusCacheCleanupService(settings=settings)
    logger.info(
        "Corpus cache cleanup CLI starting",
        extra={"event": "corpus_cache_cleanup_cli_started"},
    )
    try:
        result = service.clean()
    except CorpusCacheCleanupError as exc:
        logger.error(
            "Corpus cache cleanup CLI failed",
            extra={
                "event": "corpus_cache_cleanup_cli_failed",
                "error": str(exc),
            },
        )
        return 1
    logger.info(
        "Corpus cache cleanup CLI completed",
        extra={
            "event": "corpus_cache_cleanup_cli_completed",
            "deleted_files": result.deleted_files,
            "bytes_freed": result.bytes_freed,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

