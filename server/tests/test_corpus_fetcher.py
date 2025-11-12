from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher


def _make_transport(data: bytes) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": str(len(data)), "ETag": ""})
        if request.method == "GET":
            # Range handling is emulated by precomputing the expected start via
            # the temporary file size; since the transport cannot access local
            # state, tests that need resume will supply a custom stream.
            part = data
            status = 200
            headers = {"Content-Length": str(len(part))}
            return httpx.Response(status, content=part, headers=headers)
        return httpx.Response(405)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("resume", [False, True])
def test_fetcher_download_and_cache_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    payload = b"hello world" * 100
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    transport = _make_transport(payload)
    client = httpx.Client(transport=transport)

    def _head(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        return client.head(url, headers=headers, timeout=timeout)

    @contextmanager
    def _stream(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> Iterator[httpx.Response]:
        # Emulate Range behavior deterministically by inspecting any
        # pre-existing temporary file created by the fetcher.
        fid = url.rsplit("/", 1)[-1]
        cache_path = cache / f"{fid}.txt"
        tmp = cache_path.with_suffix(".tmp")
        start = tmp.stat().st_size if tmp.exists() else 0
        part = payload[start:]
        status = 206 if start > 0 else 200
        headers2 = {"Content-Length": str(len(part))}
        req = httpx.Request("GET", url, headers=headers)
        resp = httpx.Response(status, content=part, headers=headers2, request=req)
        try:
            yield resp
        finally:
            pass

    monkeypatch.setattr(httpx, "head", _head)
    monkeypatch.setattr(httpx, "stream", _stream)

    fid = "deadbeef"
    cache_path = cache / f"{fid}.txt"
    if resume:
        tmp = cache_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(payload[:50])

    out = f.fetch(fid)
    assert out == cache_path
    assert out.read_bytes() == payload

    out2 = f.fetch(fid)
    assert out2 == out


def test_fetcher_uses_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)
    fid = "cafebabe"
    cache_path = cache / f"{fid}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached", encoding="utf-8")

    # If HEAD is invoked, raise to fail the test
    def _head(_url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
        raise AssertionError("HEAD should not be called on cache hit")

    monkeypatch.setattr(httpx, "head", _head)

    out = f.fetch(fid)
    assert out == cache_path
    assert out.read_text(encoding="utf-8") == "cached"
