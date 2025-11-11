from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.artifacts import ArtifactListResponse
from model_trainer.core.config.settings import Settings


def _client_with_root(root: Path) -> TestClient:
    os.environ["APP__ARTIFACTS_ROOT"] = str(root)
    return TestClient(create_app(Settings()))


def test_artifacts_download_range_open_ended(tmp_path: Path) -> None:
    # Arrange: file of length 6
    base = tmp_path / "artifacts" / "models" / "r1"
    base.mkdir(parents=True)
    data = b"abcdef"
    (base / "file.bin").write_bytes(data)
    client = _client_with_root(tmp_path / "artifacts")

    # Act: request bytes from offset 2 to end (no end specified)
    r: httpx.Response = client.get(
        "/artifacts/models/r1/download",
        params={"path": "file.bin"},
        headers={"Range": "bytes=2-"},
    )

    # Assert
    assert r.status_code == 206
    assert r.content == data[2:]
    assert r.headers["Content-Range"] == f"bytes 2-{len(data)-1}/{len(data)}"


def test_artifacts_download_invalid_range_format(tmp_path: Path) -> None:
    # Arrange
    base = tmp_path / "artifacts" / "models" / "r2"
    base.mkdir(parents=True)
    (base / "file.bin").write_bytes(b"xyz")
    client = _client_with_root(tmp_path / "artifacts")

    # Act: invalid numeric values in range header
    r: httpx.Response = client.get(
        "/artifacts/models/r2/download",
        params={"path": "file.bin"},
        headers={"Range": "bytes=abc-def"},
    )

    # Assert
    assert r.status_code == 416


def test_artifacts_list_skips_directories(tmp_path: Path) -> None:
    # Arrange: create an empty subdirectory and a file; the list should only include files
    base = tmp_path / "artifacts" / "models" / "r3"
    (base / "empty").mkdir(parents=True)
    (base / "f.bin").write_bytes(b"00")
    client = _client_with_root(tmp_path / "artifacts")

    # Act
    r = client.get("/artifacts/models/r3")

    # Assert: directories are skipped, only file appears
    assert r.status_code == 200
    files = set(ArtifactListResponse.model_validate_json(r.text).files)
    assert "f.bin" in files and "empty" not in files
