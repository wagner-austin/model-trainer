from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher

_T = TypeVar("_T")


def _get_log_extra(record: logging.LogRecord, key: str, expected_type: type[_T]) -> _T | None:
    """Type-safe accessor for log record extra fields."""
    val: object = getattr(record, key, None)
    if val is None:
        return None
    if not isinstance(val, expected_type):
        return None
    return val


def _has_log_extra(record: logging.LogRecord, key: str) -> bool:
    """Check if a log record has a specific extra field."""
    if not hasattr(record, key):
        return False
    val: object = getattr(record, key, None)
    return val is not None


def test_fetcher_logs_cache_hit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that cache hits are logged."""
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)
    fid = "cached_file"
    cache_path = cache / f"{fid}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        _ = f.fetch(fid)

    assert any("Corpus cache hit" in record.message for record in caplog.records)
    # Verify structured logging extra fields
    record = next(r for r in caplog.records if "Corpus cache hit" in r.message)
    assert _get_log_extra(record, "file_id", str) == fid
    assert _get_log_extra(record, "path", str) == str(cache_path)


def test_fetcher_logs_fetch_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that fetch start is logged with structured fields."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": str(len(payload))}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request("GET", url, headers=headers)
        yield httpx.Response(200, content=payload, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    with caplog.at_level(logging.INFO):
        _ = f.fetch("test_file")

    assert any(
        "Starting corpus fetch from data bank" in record.message for record in caplog.records
    )
    record = next(r for r in caplog.records if "Starting corpus fetch" in r.message)
    assert _get_log_extra(record, "file_id", str) == "test_file"
    assert _get_log_extra(record, "api_url", str) == "http://test"


def test_fetcher_logs_head_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that HEAD request is logged."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": str(len(payload))}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request("GET", url, headers=headers)
        yield httpx.Response(200, content=payload, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    with caplog.at_level(logging.INFO):
        _ = f.fetch("test_file")

    assert any("Sending HEAD request to data bank" in record.message for record in caplog.records)
    assert any("HEAD request successful" in record.message for record in caplog.records)


def test_fetcher_logs_download_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that download start is logged with size."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": str(len(payload))}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request("GET", url, headers=headers)
        yield httpx.Response(200, content=payload, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    with caplog.at_level(logging.INFO):
        _ = f.fetch("test_file")

    assert any("Starting corpus download" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Starting corpus download" in r.message)
    assert _get_log_extra(record, "expected_size", int) == len(payload)


def test_fetcher_logs_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that successful completion is logged with elapsed time."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": str(len(payload))}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request("GET", url, headers=headers)
        yield httpx.Response(200, content=payload, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    with caplog.at_level(logging.INFO):
        _ = f.fetch("test_file")

    assert any("Corpus fetch completed successfully" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Corpus fetch completed successfully" in r.message)
    assert _get_log_extra(record, "file_id", str) == "test_file"
    assert _get_log_extra(record, "size", int) == len(payload)
    assert _has_log_extra(record, "elapsed_seconds")


def test_fetcher_logs_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that resume is logged with offset."""
    payload = b"test data"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": str(len(payload))}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        fid = url.rsplit("/", 1)[-1]
        cache_path = cache / f"{fid}.txt"
        tmp = cache_path.with_suffix(".tmp")
        start = tmp.stat().st_size if tmp.exists() else 0
        part = payload[start:]
        req = httpx.Request("GET", url, headers=headers)
        yield httpx.Response(200 if start == 0 else 206, content=part, request=req)

    # Create partial temp file
    fid = "resume_file"
    cache_path = cache / f"{fid}.txt"
    tmp = cache_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(payload[:4])

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    with caplog.at_level(logging.INFO):
        _ = f.fetch(fid)

    assert any("Resuming partial download" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Resuming partial download" in r.message)
    assert _get_log_extra(record, "resume_from", int) == 4


def test_fetcher_logs_size_mismatch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that size mismatch is logged as error."""
    payload = b"small"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        # Claim bigger size
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": "100"}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request("GET", url, headers=headers)
        yield httpx.Response(200, content=payload, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        _ = f.fetch("bad_file")

    assert any("Downloaded file size mismatch" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Downloaded file size mismatch" in r.message)
    assert _get_log_extra(record, "expected_size", int) == 100
    assert _get_log_extra(record, "actual_size", int) == len(payload)
