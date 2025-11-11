from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings


class _BinReader:
    def __init__(self: _BinReader, data: bytes, chunk: int) -> None:
        self._data = data
        self._chunk = chunk
        self._pos = 0
        self._reads = 0

    def __enter__(self: _BinReader) -> _BinReader:
        return self

    def __exit__(
        self: _BinReader,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> Literal[False]:
        return False

    def read(self: _BinReader, size: int) -> bytes:
        # First read returns at most chunk bytes, then simulate EOF early
        if self._reads >= 1:
            return b""
        self._reads += 1
        end = min(self._pos + min(size, self._chunk), len(self._data))
        out = self._data[self._pos : end]
        self._pos = end
        return out

    def seek(self: _BinReader, offset: int, whence: int = 0) -> int:
        # Support f.seek(start)
        if whence == 0:
            self._pos = offset
        return self._pos


def test_artifacts_range_stream_early_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange a file and app
    base = tmp_path / "artifacts" / "models" / "r1"
    base.mkdir(parents=True)
    data = b"abcdef"
    (base / "f.bin").write_bytes(data)
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    app = create_app(Settings())

    client = TestClient(app)
    r: httpx.Response = client.get(
        "/artifacts/models/r1/download",
        params={"path": "f.bin"},
        headers={"Range": "bytes=0-10"},
    )
    # Requesting beyond EOF triggers early break; server streams available bytes
    assert r.status_code == 206
    assert r.content == data
    assert r.headers["Content-Range"].startswith("bytes 0-10/")
