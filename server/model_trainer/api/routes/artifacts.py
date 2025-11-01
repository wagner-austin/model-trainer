from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from ...core.config.settings import Settings
from ...core.infra.paths import artifacts_root
from ...core.logging.types import LoggingExtra
from ...core.services.container import ServiceContainer
from ..schemas.artifacts import ArtifactListResponse


def _list_artifacts_impl(
    container: ServiceContainer,
    settings: Settings,
    kind: Literal["tokenizers", "models"],
    item_id: str,
) -> ArtifactListResponse:
    base = artifacts_root(settings) / kind / item_id
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail="artifact not found")
    files: list[str] = []
    for p in base.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(base))
            files.append(rel)
    extra: LoggingExtra = {
        "event": "artifacts_list",
        "kind": kind,
        "item_id": item_id,
        "count": len(files),
    }
    container.logging.adapter(category="api", service="artifacts").info(
        "artifacts list", extra=extra
    )
    return ArtifactListResponse(kind=kind, item_id=item_id, files=files)


def _download_impl(
    container: ServiceContainer,
    settings: Settings,
    kind: Literal["tokenizers", "models"],
    item_id: str,
    request: Request,
    path: str,
) -> StreamingResponse | FileResponse:
    base = artifacts_root(settings) / kind / item_id
    target = _resolve_target(base, path)
    extra2: LoggingExtra = {
        "event": "artifacts_download",
        "kind": kind,
        "item_id": item_id,
        "path": path,
    }
    container.logging.adapter(category="api", service="artifacts").info(
        "artifacts download", extra=extra2
    )
    range_header = request.headers.get("range")
    file_size = target.stat().st_size
    if range_header and range_header.startswith("bytes="):
        try:
            range_spec = range_header.replace("bytes=", "").strip()
            start_s, end_s = [*range_spec.split("-", 1), ""][:2]
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            if start < 0 or end < start or start >= file_size:
                raise ValueError("invalid range")
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="invalid range") from None
        chunk_size = 8192
        length = end - start + 1

        def iter_file() -> Iterator[bytes]:
            with open(target, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        return StreamingResponse(
            iter_file(), status_code=206, headers=headers, media_type="application/octet-stream"
        )
    return _full_file_response(target)


def _resolve_target(base: Path, rel_path: str) -> Path:
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail="artifact not found")
    target = (base / rel_path).resolve()
    try:
        base_resolved = base.resolve()
    except OSError:
        raise HTTPException(status_code=500, detail="failed to resolve artifact path") from None
    if base_resolved not in target.parents and target != base_resolved:
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return target


def _full_file_response(target: Path) -> FileResponse:
    return FileResponse(str(target), headers={"Accept-Ranges": "bytes"})


def build_router(container: ServiceContainer) -> APIRouter:
    router = APIRouter()
    settings: Settings = container.settings

    def list_artifacts(kind: Literal["tokenizers", "models"], item_id: str) -> ArtifactListResponse:
        return _list_artifacts_impl(container, settings, kind, item_id)

    def download_artifact(
        kind: Literal["tokenizers", "models"],
        item_id: str,
        request: Request,
        path: str = Query(..., description="Relative path under the artifact directory"),
    ) -> Response:
        return _download_impl(container, settings, kind, item_id, request, path)

    router.add_api_route(
        "/artifacts/{kind}/{item_id}",
        list_artifacts,
        methods=["GET"],
        response_model=ArtifactListResponse,
    )
    router.add_api_route(
        "/artifacts/{kind}/{item_id}/download",
        download_artifact,
        methods=["GET"],
    )
    return router
