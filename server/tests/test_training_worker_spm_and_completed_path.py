from __future__ import annotations

import os
from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import training_worker as tw


def test_training_worker_spm_artifact_and_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Environment roots
    artifacts = tmp_path / "artifacts"
    runs = tmp_path / "runs"
    logs = tmp_path / "logs"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    os.environ["APP__RUNS_ROOT"] = str(runs)
    os.environ["APP__LOGS_ROOT"] = str(logs)

    # Provide sentencepiece tokenizer artifact (spm) so worker loads tok_spm path
    tok_id = "tok-spm"
    tok_dir = artifacts / "tokenizers" / tok_id
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "tokenizer.model").write_bytes(b"\x00\x01mock")
    (tok_dir / "tokenizer.vocab").write_text("[UNK]\t0\nA\t1\n", encoding="utf-8")

    # Minimal corpus
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")

    # Fake redis client
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(tw, "_redis_client", lambda: fake)

    # Stub ServiceContainer to return a backend that completes immediately
    class _Backend:
        def prepare(
            self: _Backend,
            cfg: ModelTrainConfig,
            settings: Settings,
            *,
            tokenizer: object,
        ) -> object:
            return object()

        class _Res:
            cancelled = False
            loss = 0.9
            perplexity = 1.5
            steps = 10

        def train(
            self: _Backend,
            cfg: ModelTrainConfig,
            settings: Settings,
            *,
            run_id: str,
            heartbeat: object,
            cancelled: object,
            prepared: object,
            progress: object,
        ) -> _Backend._Res:
            return _Backend._Res()

        def save(self: _Backend, prepared: object, out_dir: str) -> str:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "weights.bin").write_bytes(b"\x00mock")
            return out_dir

    class _Reg:
        def get(self: _Reg, name: str) -> _Backend:
            return _Backend()

    class _C:
        def __init__(self: _C) -> None:
            self.model_registry = _Reg()

        @staticmethod
        def from_settings(_: Settings) -> _C:
            return _C()

    monkeypatch.setattr(tw, "ServiceContainer", _C)

    payload: TrainJobPayload = {
        "run_id": "run-complete",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": tok_id,
            "corpus_file_id": "deadbeef",
        },
    }

    # Stub fetcher to map file id to local corpus
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return corpus

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    tw.process_train_job(payload)
    assert fake.get("runs:status:run-complete") == "completed"
    assert fake.get("runs:msg:run-complete") == "Training completed"
    out_dir = artifacts / "models" / "run-complete"
    assert (out_dir / "weights.bin").exists()
