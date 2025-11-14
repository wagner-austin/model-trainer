from __future__ import annotations

import io
from pathlib import Path

import pytest
from model_trainer.core.services.data.artifact_uploader import (
    ArtifactUploader,
    ArtifactUploadError,
)
from model_trainer.core.services.data.data_bank_client import DataBankClientError


def test_artifact_uploader_config_missing(tmp_path: Path) -> None:
    up = ArtifactUploader(api_url="", api_key="")
    with pytest.raises(ArtifactUploadError):
        up.upload_dir(tmp_path, name="n", request_id="rid")


def test_artifact_uploader_missing_directory(tmp_path: Path) -> None:
    """Directory path that does not exist should raise ArtifactUploadError."""
    base = tmp_path / "missing-dir"
    uploader = ArtifactUploader(api_url="http://db.local", api_key="secret")
    with pytest.raises(ArtifactUploadError):
        uploader.upload_dir(base, name="model-missing", request_id="run-missing")


def test_artifact_uploader_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a small directory to archive
    base = tmp_path / "model"
    (base / "sub").mkdir(parents=True, exist_ok=True)
    (base / "weights.bin").write_bytes(b"\x00\x01mock")
    (base / "sub" / "cfg.json").write_text("{}", encoding="utf-8")

    # Monkeypatch DataBankClient.upload inside the module to avoid network
    import model_trainer.core.services.data.artifact_uploader as au

    class _Res:
        file_id = "deadbeef"
        size = 10
        sha256 = "deadbeef"
        content_type = "application/x-tar"
        created_at = None

    class _C:
        def __init__(self: _C, *args: object, **kwargs: object) -> None:
            pass

        def upload(
            self: _C,
            stream: io.BufferedReader,
            *,
            filename: str,
            content_type: str,
            request_id: str,
        ) -> _Res:
            data = stream.read(64)
            assert isinstance(data, bytes)
            assert filename.endswith(".tar")
            assert content_type == "application/x-tar"
            assert request_id == "run1"
            return _Res()

    monkeypatch.setattr(au, "DataBankClient", _C)
    uploader = ArtifactUploader(api_url="http://db.local", api_key="secret")
    fid = uploader.upload_dir(base, name="model-run1", request_id="run1")
    assert fid == "deadbeef"


def test_artifact_uploader_invalid_file_id_and_unlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Directory to archive
    base = tmp_path / "model2"
    base.mkdir(parents=True, exist_ok=True)
    (base / "weights.bin").write_bytes(b"data")

    import model_trainer.core.services.data.artifact_uploader as au

    class _ResBad:
        file_id = "  "  # invalid: triggers defensive branch
        size = None
        sha256 = None
        content_type = None
        created_at = None

    class _ClientBad:
        def __init__(self: _ClientBad, *args: object, **kwargs: object) -> None:
            pass

        def upload(
            self: _ClientBad,
            stream: io.BufferedReader,
            *,
            filename: str,
            content_type: str,
            request_id: str,
        ) -> _ResBad:
            # Consume some bytes to exercise tar streaming, but ignore contents.
            _ = stream.read(32)
            return _ResBad()

    def _unlink_fail(path: str) -> None:
        raise OSError("unlink-failed")

    monkeypatch.setattr(au, "DataBankClient", _ClientBad)
    monkeypatch.setattr(
        "model_trainer.core.services.data.artifact_uploader.os.unlink", _unlink_fail
    )

    uploader = ArtifactUploader(api_url="http://db.local", api_key="secret")
    with pytest.raises(ArtifactUploadError):
        uploader.upload_dir(base, name="model-run2", request_id="run2")

    # Ensure the unlink failure path logs a warning
    joined = "\n".join(caplog.messages)
    assert "failed to remove temp tar" in joined


def test_artifact_uploader_wraps_data_bank_client_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "model3"
    base.mkdir(parents=True, exist_ok=True)
    (base / "weights.bin").write_bytes(b"data")

    import model_trainer.core.services.data.artifact_uploader as au

    class _ClientError:
        def __init__(self: _ClientError, *args: object, **kwargs: object) -> None:
            pass

        def upload(
            self: _ClientError,
            stream: io.BufferedReader,
            *,
            filename: str,
            content_type: str,
            request_id: str,
        ) -> object:
            _ = stream.read(8)
            raise DataBankClientError("boom")

    monkeypatch.setattr(au, "DataBankClient", _ClientError)
    uploader = ArtifactUploader(api_url="http://db.local", api_key="secret")
    with pytest.raises(ArtifactUploadError):
        uploader.upload_dir(base, name="model-run3", request_id="run3")
