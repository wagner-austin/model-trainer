from __future__ import annotations

import os
import shutil
from pathlib import Path

import fakeredis
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.core.services.container import ServiceContainer
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job


def test_sentencepiece_orchestrator_fails_without_cli(monkeypatch: MonkeyPatch) -> None:
    # Force CLI to be unavailable regardless of host
    def _which(name: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _which, raising=True)
    app = create_app(Settings())
    container: ServiceContainer = app.state.container
    container.redis = fakeredis.FakeRedis(decode_responses=True)
    client = TestClient(app)
    body = {
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    r = client.post("/tokenizers/train", json=body)
    assert r.status_code == 400


def test_sentencepiece_worker_trains_and_writes_artifacts_with_mocked_cli(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Pretend CLI exists
    def _which(name: str) -> str:
        return "spm_mock"

    monkeypatch.setattr(shutil, "which", _which, raising=True)
    # Mock subprocess in spm_backend to simulate CLI behavior

    class _Proc:
        def __init__(self: _Proc, stdout: str) -> None:
            self.stdout: str = stdout

    def _fake_run(
        args: list[str] | tuple[str, ...],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        input: str | None = None,
        **kwargs: object,
    ) -> _Proc:
        cmd = str(args[0])
        if "spm_train" in cmd:
            prefix = next(a.split("=", 1)[1] for a in args if str(a).startswith("--model_prefix="))
            model_path = Path(prefix + ".model")
            vocab_path = Path(prefix + ".vocab")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"\x00\x01mock")
            vocab_path.write_text("[UNK]\t0\nA\t0\nB\t0\n", encoding="utf-8")
            return _Proc("")
        if "spm_encode" in cmd:
            return _Proc("1 2 3\n")
        if "spm_decode" in cmd:
            return _Proc(input or "")
        return _Proc("")

    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.spm_backend.subprocess.run",
        _fake_run,
        raising=True,
    )
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis.from_url",
        _fake_from_url,
    )

    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)

    # Minimal corpus
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is spm\n", encoding="utf-8")

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm",
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    # Stub fetcher to return the local corpus directory
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return corpus

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm:status") == "completed"
    out_dir = artifacts / "tokenizers" / "tok-spm"
    assert (out_dir / "tokenizer.model").exists()
    assert (out_dir / "manifest.json").exists()
