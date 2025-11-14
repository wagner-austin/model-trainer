from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest
from model_trainer.core.services.data.artifact_downloader import (
    ArtifactDownloadError,
    ArtifactDownloader,
)


def _make_tar_with_root(tmp_path: Path, root_name: str) -> Path:
    base = tmp_path / root_name
    base.mkdir()
    (base / "weights.bin").write_bytes(b"x")
    tar_path = tmp_path / f"{root_name}.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(base):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(base)
                arcname = Path(root_name) / rel
                tf.add(str(abs_path), arcname=str(arcname))
    return tar_path


def test_artifact_downloader_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tar_path = _make_tar_with_root(tmp_path, "model-run123")

    class _ClientStub:
        def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 0.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            assert file_id == "fid-123"
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(
        "model_trainer.core.services.data.artifact_downloader.DataBankClient", _ClientStub
    )

    downloader = ArtifactDownloader(api_url="http://db", api_key="k")
    target_root = tmp_path / "models"
    dest_dir = downloader.download_and_extract("fid-123", run_id="run123", target_root=target_root)

    assert dest_dir == target_root / "run123"
    assert dest_dir.exists()
    assert (dest_dir / "weights.bin").exists()


def test_artifact_downloader_config_missing(tmp_path: Path) -> None:
    downloader = ArtifactDownloader(api_url="", api_key="k")
    with pytest.raises(ArtifactDownloadError, match="configuration missing"):
        downloader.download_and_extract("fid", run_id="run", target_root=tmp_path)


def test_artifact_downloader_empty_file_id(tmp_path: Path) -> None:
    downloader = ArtifactDownloader(api_url="http://db", api_key="k")
    with pytest.raises(ArtifactDownloadError, match="file_id must be non-empty"):
        downloader.download_and_extract("   ", run_id="run", target_root=tmp_path)


def test_artifact_downloader_unexpected_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tar_path = _make_tar_with_root(tmp_path, "wrong-root")

    class _ClientStub:
        def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 0.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(
        "model_trainer.core.services.data.artifact_downloader.DataBankClient", _ClientStub
    )

    downloader = ArtifactDownloader(api_url="http://db", api_key="k")
    with pytest.raises(ArtifactDownloadError, match="unexpected archive root"):
        downloader.download_and_extract("fid", run_id="run123", target_root=tmp_path / "models")


def test_artifact_downloader_dest_dir_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tar_path = _make_tar_with_root(tmp_path, "model-run123")

    class _ClientStub:
        def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 0.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(
        "model_trainer.core.services.data.artifact_downloader.DataBankClient", _ClientStub
    )

    target_root = tmp_path / "models"
    existing = target_root / "run123"
    existing.mkdir(parents=True, exist_ok=True)

    downloader = ArtifactDownloader(api_url="http://db", api_key="k")
    with pytest.raises(ArtifactDownloadError, match="destination already exists"):
        downloader.download_and_extract("fid", run_id="run123", target_root=target_root)


def test_artifact_downloader_unexpected_layout_multiple_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build tar with two different top-level roots to trigger layout error
    base1 = tmp_path / "model-run123"
    base1.mkdir()
    (base1 / "a.bin").write_bytes(b"a")
    base2 = tmp_path / "other-root"
    base2.mkdir()
    (base2 / "b.bin").write_bytes(b"b")
    tar_path = tmp_path / "multi.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(base1):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(base1)
                arcname = Path("model-run123") / rel
                tf.add(str(abs_path), arcname=str(arcname))
        for root, _, files in os.walk(base2):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(base2)
                arcname = Path("other-root") / rel
                tf.add(str(abs_path), arcname=str(arcname))

    class _ClientStub:
        def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 0.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(
        "model_trainer.core.services.data.artifact_downloader.DataBankClient", _ClientStub
    )

    downloader = ArtifactDownloader(api_url="http://db", api_key="k")
    with pytest.raises(ArtifactDownloadError, match="unexpected archive layout"):
        downloader.download_and_extract("fid", run_id="run123", target_root=tmp_path / "models")


def test_artifact_downloader_root_not_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tar has a single root entry named model-run123 but it is a file, not a directory.
    root_name = "model-run123"
    tar_path = tmp_path / "file_root.tar"
    file_root = tmp_path / root_name
    file_root.write_bytes(b"payload")
    with tarfile.open(str(tar_path), "w") as tf:
        tf.add(str(file_root), arcname=root_name)

    class _ClientStub:
        def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 0.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(
        "model_trainer.core.services.data.artifact_downloader.DataBankClient", _ClientStub
    )

    downloader = ArtifactDownloader(api_url="http://db", api_key="k")
    with pytest.raises(ArtifactDownloadError, match="extracted model directory missing"):
        downloader.download_and_extract("fid", run_id="run123", target_root=tmp_path / "models")
