from __future__ import annotations

import json
from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.worker.training_worker import EVAL_KEY_PREFIX, process_eval_job
from pytest import MonkeyPatch


class _Backend:
    def evaluate(self: _Backend, *, run_id: str, cfg: object, settings: Settings) -> object:
        raise RuntimeError("boom")


class _ModelRegistry:
    def get(self: _ModelRegistry, name: str) -> _Backend:
        return _Backend()


class _Container:
    def __init__(self: _Container, registry: _ModelRegistry) -> None:
        self.model_registry = registry


def test_worker_eval_backend_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Fake redis (shared instance for reads in assertions)
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("redis.from_url", _from_url)

    # Manifest exists to pass earlier branch
    s = Settings()
    artifacts = Path(s.app.artifacts_root)
    run_dir = artifacts / "models" / "run-err"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "run-err",
        "model_family": "gpt2",
        "model_size": "s",
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 8,
        "steps": 0,
        "loss": 0.0,
        "learning_rate": 1e-3,
        "tokenizer_id": "tok",
        "corpus_path": str(tmp_path),
        "optimizer": "AdamW",
        "seed": 42,
        "versions": {"torch": "0", "transformers": "0", "tokenizers": "0", "datasets": "0"},
        "system": {"cpu_count": 1, "platform": "X", "platform_release": "Y", "machine": "Z"},
        "git_commit": None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Patch container factory to return backend that raises
    def _from_settings(settings: Settings) -> _Container:
        return _Container(_ModelRegistry())

    monkeypatch.setattr(
        "model_trainer.core.services.container.ServiceContainer.from_settings", _from_settings
    )

    # Now run eval and assert failure is recorded and exception propagated
    payload: EvalJobPayload = {"run_id": "run-err", "split": "validation", "path_override": None}
    with pytest.raises(RuntimeError):
        process_eval_job(payload)
    raw = fake.get(f"{EVAL_KEY_PREFIX}run-err")
    assert raw is not None
