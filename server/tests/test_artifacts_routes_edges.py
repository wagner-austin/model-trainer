from __future__ import annotations

import os
from pathlib import Path

import fakeredis
from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_trainer.api.routes import artifacts as artifacts_routes
from model_trainer.core.config.settings import Settings
from model_trainer.core.logging.service import LoggingService
from model_trainer.core.services.container import ServiceContainer
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.queue.rq_adapter import RQEnqueuer, RQSettings
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry
from model_trainer.orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from model_trainer.orchestrators.training_orchestrator import TrainingOrchestrator
from pytest import MonkeyPatch


def _container_for(settings: Settings, r: fakeredis.FakeRedis) -> ServiceContainer:
    enq = RQEnqueuer("redis://ignored", RQSettings("q", 1, 1, 1, 0, []))
    training = TrainingOrchestrator(
        settings=settings,
        redis_client=r,
        enqueuer=enq,
        model_registry=ModelRegistry(backends={}),
    )
    tokenizer = TokenizerOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
    return ServiceContainer(
        settings=settings,
        redis=r,
        rq_enqueuer=enq,
        training_orchestrator=training,
        tokenizer_orchestrator=tokenizer,
        model_registry=ModelRegistry(backends={}),
        tokenizer_registry=TokenizerRegistry(backends={}),
        logging=LoggingService.create(),
        dataset_builder=LocalTextDatasetBuilder(),
    )


def _mk_client(tmp: Path, monkeypatch: MonkeyPatch) -> tuple[TestClient, Settings]:
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp / "artifacts")
    s = Settings()

    # Fake redis
    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fakeredis.FakeRedis(decode_responses=True)

    monkeypatch.setattr("redis.from_url", _fake_from_url)
    container = _container_for(s, fakeredis.FakeRedis(decode_responses=True))
    app = FastAPI()
    app.include_router(artifacts_routes.build_router(container))
    return TestClient(app), s


def test_artifacts_list_not_found(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client, _ = _mk_client(tmp_path, monkeypatch)
    res = client.get("/artifacts/models/nope")
    assert res.status_code == 404


def test_artifacts_list_and_download_range(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client, settings = _mk_client(tmp_path, monkeypatch)
    base = Path(settings.app.artifacts_root) / "models" / "r1"
    base.mkdir(parents=True, exist_ok=True)
    f = base / "file.bin"
    data = b"abcdefg"
    f.write_bytes(data)
    # List shows file
    res = client.get("/artifacts/models/r1")
    assert res.status_code == 200
    from pydantic import BaseModel

    class _List(BaseModel):
        kind: str
        item_id: str
        files: list[str]

    parsed = _List.model_validate_json(res.text)
    assert "file.bin" in parsed.files
    # Valid range
    res2 = client.get(
        "/artifacts/models/r1/download",
        headers={"range": "bytes=2-4"},
        params={"path": "file.bin"},
    )
    assert res2.status_code == 206
    assert res2.content == data[2:5]
    assert res2.headers["Content-Range"].startswith("bytes 2-4/")
    # Invalid range
    res3 = client.get(
        "/artifacts/models/r1/download",
        headers={"range": "bytes=9-"},
        params={"path": "file.bin"},
    )
    assert res3.status_code == 416


def test_artifacts_download_invalid_path_and_not_found(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client, settings = _mk_client(tmp_path, monkeypatch)
    base = Path(settings.app.artifacts_root) / "models" / "r2"
    base.mkdir(parents=True, exist_ok=True)
    # invalid path escapes base via parent reference
    res = client.get("/artifacts/models/r2/download", params={"path": "../secret.txt"})
    assert res.status_code == 400
    # file not found under base
    res2 = client.get("/artifacts/models/r2/download", params={"path": "missing.bin"})
    assert res2.status_code == 404


def test_artifacts_download_failed_resolve(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    client, settings = _mk_client(tmp_path, monkeypatch)
    base = Path(settings.app.artifacts_root) / "models" / "r3"
    base.mkdir(parents=True, exist_ok=True)
    # Monkeypatch Path.resolve to raise OSError when called for base
    from pathlib import Path as PathAlias

    orig_resolve = PathAlias.resolve

    def _boom(self: PathAlias) -> Path:
        if str(self).endswith(str(base)):
            raise OSError("bad path")
        return orig_resolve(self)

    monkeypatch.setattr(PathAlias, "resolve", _boom)
    res = client.get("/artifacts/models/r3/download", params={"path": "file.bin"})
    assert res.status_code == 500
