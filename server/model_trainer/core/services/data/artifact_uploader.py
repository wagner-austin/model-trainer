from __future__ import annotations

import os
import tarfile
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path

from .data_bank_client import DataBankClient, DataBankClientError


class ArtifactUploadError(Exception):
    pass


@dataclass
class ArtifactUploader:
    api_url: str
    api_key: str

    def _ensure_cfg(self: "ArtifactUploader") -> None:
        if self.api_url.strip() == "" or self.api_key.strip() == "":
            raise ArtifactUploadError("data-bank-api configuration missing")

    def upload_dir(self: "ArtifactUploader", dir_path: Path, *, name: str, request_id: str) -> str:
        self._ensure_cfg()
        base = Path(dir_path)
        if not base.exists() or not base.is_dir():
            raise ArtifactUploadError("artifact directory not found")

        # Create a temporary tarball for the directory to ensure deterministic upload
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="artifact_", suffix=".tar")
        os.close(tmp_fd)
        tar_path = Path(tmp_path)
        try:
            with tarfile.open(str(tar_path), mode="w") as tf:
                # Archive the directory contents under a top-level folder named `name`
                for root, _, files in os.walk(base):
                    for fn in files:
                        abs_path = Path(root) / fn
                        rel = abs_path.relative_to(base)
                        arcname = Path(name) / rel
                        tf.add(str(abs_path), arcname=str(arcname))

            client = DataBankClient(base_url=self.api_url, api_key=self.api_key, timeout_seconds=600.0)
            with tar_path.open("rb") as f:
                res = client.upload(f, filename=f"{name}.tar", content_type="application/x-tar", request_id=request_id)
            fid = res.file_id
            if not isinstance(fid, str) or fid.strip() == "":  # pragma: no cover - defensive
                raise ArtifactUploadError("invalid file_id from data-bank-api")
            return fid
        except DataBankClientError as e:
            raise ArtifactUploadError(str(e)) from e
        finally:
            try:
                os.unlink(str(tar_path))
            except OSError as e:
                logging.getLogger(__name__).warning("failed to remove temp tar: %s", e)
