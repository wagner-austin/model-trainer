from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

from .data_bank_client import DataBankClient


class ArtifactDownloadError(Exception):
    """Raised when artifact download or extraction fails."""


@dataclass
class ArtifactDownloader:
    api_url: str
    api_key: str

    def _ensure_cfg(self: "ArtifactDownloader") -> None:
        if self.api_url.strip() == "" or self.api_key.strip() == "":
            raise ArtifactDownloadError("data-bank-api configuration missing")

    def download_and_extract(
        self: "ArtifactDownloader",
        file_id: str,
        *,
        run_id: str,
        target_root: Path,
    ) -> Path:
        """Download a model artifact tar from data-bank-api and extract it under target_root.

        The archive is expected to contain a single top-level directory named `model-{run_id}`.
        After extraction, that directory is renamed to `{run_id}` under `target_root`.
        """
        self._ensure_cfg()
        fid = file_id.strip()
        if fid == "":
            raise ArtifactDownloadError("file_id must be non-empty")

        client = DataBankClient(base_url=self.api_url, api_key=self.api_key, timeout_seconds=600.0)

        # Download to a temporary tarball
        from tempfile import mkstemp

        fd, tmp_path_str = mkstemp(prefix="model_artifact_", suffix=".tar")
        os.close(fd)
        tar_path = Path(tmp_path_str)

        try:
            client.download_to_path(fid, tar_path, request_id=run_id, verify_etag=True)

            dest_parent = target_root
            dest_parent.mkdir(parents=True, exist_ok=True)

            with tarfile.open(str(tar_path), mode="r") as tf:
                members = tf.getmembers()
                root_names: set[str] = set()
                for m in members:
                    root_names.add(m.name.split("/", 1)[0])
                if len(root_names) != 1:
                    raise ArtifactDownloadError("unexpected archive layout")
                (root_name,) = root_names
                expected_root = f"model-{run_id}"
                if root_name != expected_root:
                    raise ArtifactDownloadError(
                        f"unexpected archive root '{root_name}', expected '{expected_root}'"
                    )
                dest_dir = dest_parent / run_id
                if dest_dir.exists():
                    raise ArtifactDownloadError(f"destination already exists: {dest_dir}")
                tf.extractall(dest_parent)

            src_dir = dest_parent / f"model-{run_id}"
            if not src_dir.exists() or not src_dir.is_dir():
                raise ArtifactDownloadError("extracted model directory missing after unpack")
            src_dir.rename(dest_dir)
            return dest_dir
        finally:
            if tar_path.exists():
                os.unlink(str(tar_path))
