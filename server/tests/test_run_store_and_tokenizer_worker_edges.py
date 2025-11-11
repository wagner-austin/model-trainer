from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.infra.storage.run_store import RunStore
from model_trainer.worker import tokenizer_worker as tkw


def test_run_store_manifest_write_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rs = RunStore(artifacts_root=str(tmp_path / "artifacts"))
    # Force open to raise OSError only for manifest path
    # Instead of patching open overloads, raise from json.dump to hit error path
    import json as _json

    def _fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(_json, "dump", _fail_dump)
    run_id = rs.create_run("gpt2", "small")
    assert isinstance(run_id, str) and run_id


def test_tokenizer_worker_sentencepiece_missing_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(tkw, "_redis_client", lambda: fake)
    # Artifacts root
    monkeypatch.setenv("APP__ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    payload: TokenizerTrainPayload = {
        "tokenizer_id": "t1",
        "method": "sentencepiece",
        "vocab_size": 100,
        "min_frequency": 1,
        "corpus_path": str(tmp_path),
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    import shutil

    def _which_none(name: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _which_none)
    tkw.process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:t1:status") == "failed"
