from __future__ import annotations

from pathlib import Path
from typing import Final

import httpx


class CorpusFetcher:
    def __init__(self, api_url: str, api_key: str, cache_dir: Path) -> None:
        self._api_url: Final[str] = api_url.rstrip("/")
        self._api_key: Final[str] = api_key
        self._cache_dir: Final[Path] = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, file_id: str) -> Path:
        cache_path = self._cache_dir / f"{file_id}.txt"
        if cache_path.exists():
            return cache_path

        headers = {"X-API-Key": self._api_key}
        url = f"{self._api_url}/files/{file_id}"

        head_resp = httpx.head(url, headers=headers, timeout=30.0)
        head_resp.raise_for_status()
        # HEAD headers are case-insensitive, normalize
        hdrs = {k.lower(): v for (k, v) in head_resp.headers.items()}
        expected_size = int(hdrs.get("content-length", "0"))

        temp_path = cache_path.with_suffix(".tmp")
        start = temp_path.stat().st_size if temp_path.exists() else 0
        if start > 0:
            headers["Range"] = f"bytes={start}-"

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
            raise RuntimeError(
                f"Size mismatch: expected {expected_size}, got {temp_path.stat().st_size}"
            )

        temp_path.replace(cache_path)
        return cache_path
