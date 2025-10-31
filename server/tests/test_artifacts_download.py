from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.api.schemas.artifacts import ArtifactListResponse
from model_trainer.core.config.settings import Settings


def _app_with_artifacts_root(root: Path) -> TestClient:
    os.environ["APP__ARTIFACTS_ROOT"] = str(root)
    app = create_app(Settings())
    return TestClient(app)


def test_artifacts_list_and_download_full_and_range(tmp_path: Path) -> None:
    # Arrange
    artifacts_dir = tmp_path / "artifacts" / "models" / "run-1"
    artifacts_dir.mkdir(parents=True)
    file_path = artifacts_dir / "sample.bin"
    data = b"abcdefghijklmnopqrstuvwxyz"
    file_path.write_bytes(data)
    client = _app_with_artifacts_root(tmp_path / "artifacts")

    # List
    r: httpx.Response = client.get("/artifacts/models/run-1")
    assert r.status_code == 200
    files = ArtifactListResponse.model_validate_json(r.text).files
    assert "sample.bin" in files

    # Full download
    r2: httpx.Response = client.get(
        "/artifacts/models/run-1/download", params={"path": "sample.bin"}
    )
    assert r2.status_code == 200
    assert r2.content == data
    assert r2.headers["Accept-Ranges"] == "bytes"

    # Range download
    headers = {"Range": "bytes=5-9"}
    r3: httpx.Response = client.get(
        "/artifacts/models/run-1/download", params={"path": "sample.bin"}, headers=headers
    )
    assert r3.status_code == 206
    assert r3.content == data[5:10]
    assert r3.headers["Content-Range"] == f"bytes 5-9/{len(data)}"
