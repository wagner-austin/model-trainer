from __future__ import annotations

import os
from pathlib import Path

import fakeredis
import pytest
import redis
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import training_worker as tw


def test_emit_events_helpers_publish() -> None:
    # Use a real FakeRedis to exercise publish path without failures
    r = fakeredis.FakeRedis(decode_responses=True)
    run_id = "r-1"
    cfg = ModelTrainConfig(
        model_family="gpt2",
        model_size="small",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        tokenizer_id="tok",
        corpus_path="/dev/null",
    )
    # Does not raise
    tw._emit_started_event(r, run_id, 123, cfg, threads=2)
    tw._emit_progress_event(r, run_id, 123, epoch=1, total_epochs=1, step=1, loss=1.23)
    tw._emit_completed_event(r, run_id, 123, loss=0.9, perplexity=1.5, artifact_path="/a")
    tw._emit_failed_event(r, run_id, 123, message="x", status="failed")


def test_process_train_job_sets_status_message_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: artifacts and payload
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    os.environ["APP__RUNS_ROOT"] = str(tmp_path / "runs")
    os.environ["APP__LOGS_ROOT"] = str(tmp_path / "logs")

    payload: TrainJobPayload = {
        "run_id": "run-exc",
        "user_id": 7,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": "tok",
            "corpus_file_id": "deadbeef",
        },
    }
    (tmp_path / "corpus").mkdir()
    # Stub fetcher to return local corpus dir
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return tmp_path / "corpus"

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    (tmp_path / "corpus" / "a.txt").write_text("hello\n", encoding="utf-8")

    # Fake redis and force .set on message key to raise, so we hit the nested except and re-raise
    client = fakeredis.FakeRedis(decode_responses=True)

    orig_set = client.set

    def _patched_set(key: str, value: str) -> object:
        if key.startswith(tw.MSG_KEY_PREFIX):
            raise TypeError("simulated set error")
        return orig_set(key, value)

    monkeypatch.setattr(client, "set", _patched_set)

    def _fake_client() -> redis.Redis[str]:
        return client

    monkeypatch.setattr(tw, "_redis_client", _fake_client)

    # Force an exception by making container/model registry lookups blow up
    class _C:
        @staticmethod
        def from_settings(_: Settings) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(tw, "ServiceContainer", _C)

    with pytest.raises(RuntimeError):
        tw.process_train_job(payload)
