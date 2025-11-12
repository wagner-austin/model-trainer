from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher


def test_fetcher_head_404_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(404, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    with pytest.raises(httpx.HTTPStatusError):
        _ = f.fetch("deadbeef")


def test_fetcher_get_401_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": "10"}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request(method, url, headers=headers)
        yield httpx.Response(401, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)
    with pytest.raises(httpx.HTTPStatusError):
        _ = f.fetch("deadbeef")


def test_fetcher_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)
    payload = b"abcde"

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Length": "10"})

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request(method, url, headers=headers)
        yield httpx.Response(200, content=payload, request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)
    with pytest.raises(RuntimeError):
        _ = f.fetch("deadbeef")


def test_fetcher_resume_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)
    payload = b"abc"

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        # Expect total 6 bytes, but we'll only provide 3 including resume
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={"Content-Length": "6"}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        # Simulate Range resume but return empty to force mismatch
        req = httpx.Request(method, url, headers=headers)
        yield httpx.Response(200, content=b"", request=req)

    # Seed partial temp file to trigger Range branch
    fid = "resume"
    cache_path = tmp_path / f"{fid}.txt"
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_bytes(payload)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)
    with pytest.raises(RuntimeError):
        _ = f.fetch(fid)


def test_fetcher_head_without_content_length_allows_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        # No Content-Length header returned
        req = httpx.Request("HEAD", url, headers=headers)
        return httpx.Response(200, headers={}, request=req)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        req = httpx.Request(method, url, headers=headers)
        # Empty body
        yield httpx.Response(200, content=b"", request=req)

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)
    out = f.fetch("zero")
    assert out.exists() and out.stat().st_size == 0
