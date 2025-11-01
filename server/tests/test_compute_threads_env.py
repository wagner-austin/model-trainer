from __future__ import annotations

import os

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.compute import LocalCPUProvider


def test_local_cpu_provider_uses_configured_threads(monkeypatch: object) -> None:
    os.environ["APP__THREADS"] = "2"
    settings = Settings()
    threads = settings.app.threads if settings.app.threads > 0 else 1
    env = LocalCPUProvider(threads_count=threads).env()
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
