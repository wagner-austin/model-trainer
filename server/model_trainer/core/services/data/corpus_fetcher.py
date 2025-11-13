from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import httpx

_logger: Final[logging.Logger] = logging.getLogger(__name__)


class CorpusFetcher:
    """Downloads corpus files from data-bank-api with caching and structured logging."""

    def __init__(self, api_url: str, api_key: str, cache_dir: Path) -> None:
        self._api_url: Final[str] = api_url.rstrip("/")
        self._api_key: Final[str] = api_key
        self._cache_dir: Final[Path] = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, file_id: str) -> Path:
        """Fetch a corpus file from the data bank API, with caching.

        Args:
            file_id: The unique identifier for the corpus file.

        Returns:
            Path to the cached corpus file.

        Raises:
            httpx.HTTPStatusError: For HTTP errors (404, 401, 503, etc).
            httpx.TimeoutException: If request times out.
            httpx.ConnectError: If connection fails.
            RuntimeError: If downloaded file size doesn't match expected size.
        """
        cache_path = self._cache_dir / f"{file_id}.txt"
        if cache_path.exists():
            _logger.info("Corpus cache hit", extra={"file_id": file_id, "path": str(cache_path)})
            return cache_path

        _logger.info(
            "Starting corpus fetch from data bank",
            extra={"file_id": file_id, "api_url": self._api_url},
        )
        start_time = time.time()

        headers = {"X-API-Key": self._api_key}
        url = f"{self._api_url}/files/{file_id}"

        _logger.info("Sending HEAD request to data bank", extra={"file_id": file_id, "url": url})
        head_resp = httpx.head(url, headers=headers, timeout=30.0)
        head_resp.raise_for_status()

        # HEAD headers are case-insensitive, normalize
        hdrs = {k.lower(): v for (k, v) in head_resp.headers.items()}
        expected_size = int(hdrs.get("content-length", "0"))
        _logger.info(
            "HEAD request successful",
            extra={"file_id": file_id, "expected_size": expected_size},
        )

        temp_path = cache_path.with_suffix(".tmp")
        start = temp_path.stat().st_size if temp_path.exists() else 0
        if start > 0:
            headers["Range"] = f"bytes={start}-"
            _logger.info(
                "Resuming partial download",
                extra={"file_id": file_id, "resume_from": start},
            )

        _logger.info(
            "Starting corpus download",
            extra={"file_id": file_id, "url": url, "expected_size": expected_size},
        )
        with httpx.stream("GET", url, headers=headers, timeout=600.0) as resp:
            resp.raise_for_status()
            if start > 0:
                with temp_path.open("ab") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
            else:
                with temp_path.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        if temp_path.stat().st_size != expected_size:
            _logger.error(
                "Downloaded file size mismatch",
                extra={
                    "file_id": file_id,
                    "expected_size": expected_size,
                    "actual_size": temp_path.stat().st_size,
                },
            )
            raise RuntimeError(
                f"Size mismatch: expected {expected_size}, got {temp_path.stat().st_size}"
            )

        temp_path.replace(cache_path)
        elapsed = time.time() - start_time
        _logger.info(
            "Corpus fetch completed successfully",
            extra={
                "file_id": file_id,
                "size": expected_size,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        return cache_path
