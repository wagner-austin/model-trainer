from __future__ import annotations

import os
import tarfile
from pathlib import Path

import fakeredis
import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.infra.paths import model_dir as _model_dir
from model_trainer.core.services.model.backends.gpt2 import (
    GPT2TrainConfig,
    prepare_gpt2_with_handle,
    train_prepared_gpt2,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker.training_worker import (
    EVAL_KEY_PREFIX,
    process_eval_job,
)
from pytest import MonkeyPatch


def test_eval_job_success(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Use fake redis
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("redis.from_url", _fake_from_url)

    # Prepare artifacts and tokenizer
    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    settings_for_train = Settings()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")

    tok_id = "tok-eval"
    tok_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    # Train and persist a tiny model for run_id
    run_id = "run-eval"
    cfg = GPT2TrainConfig(
        model_family="gpt2",
        model_size="small",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        tokenizer_id=tok_id,
        corpus_path=str(corpus),
    )
    tok_handle = BPEBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = prepare_gpt2_with_handle(tok_handle, cfg)

    # heartbeat/cancel no-ops
    def _hb(_: float) -> None:  # pragma: no cover - trivial
        pass

    def _cancelled() -> bool:  # pragma: no cover - trivial
        return False

    _ = train_prepared_gpt2(
        prepared,
        cfg,
        settings_for_train,
        run_id=run_id,
        redis_hb=_hb,
        cancelled=_cancelled,
    )

    # Create a tarball matching ArtifactUploader layout and remove local model dir
    run_dir = _model_dir(settings_for_train, run_id)
    name = f"model-{run_id}"
    tar_root = tmp_path / "db"
    tar_root.mkdir()
    tar_path = tar_root / f"{run_id}.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(run_dir):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(run_dir)
                arcname = Path(name) / rel
                tf.add(str(abs_path), arcname=str(arcname))

    import shutil

    shutil.rmtree(run_dir)

    # Configure data-bank and pointer
    os.environ["APP__DATA_BANK_API_URL"] = "http://data-bank-api.local"
    os.environ["APP__DATA_BANK_API_KEY"] = "secret-key"
    file_id = "fid-eval-1"
    fake.set(f"runs:artifact:{run_id}:file_id", file_id)

    # Stub DataBankClient used by ArtifactDownloader
    from model_trainer.core.services.data import artifact_downloader as ad_mod

    class _ClientStub:
        def __init__(
            self: _ClientStub, base_url: str, api_key: str, timeout_seconds: float = 0.0
        ) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self: _ClientStub,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            assert file_id == "fid-eval-1"
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(ad_mod, "DataBankClient", _ClientStub)

    # Now process eval job using the worker entry
    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    process_eval_job(payload)
    raw = fake.get(f"{EVAL_KEY_PREFIX}{run_id}")
    assert raw is not None
    # Ensure status completed and metrics are present
    from pydantic import BaseModel

    class _Eval(BaseModel):
        status: str
        split: str
        loss: float | None = None
        ppl: float | None = None
        artifact: str | None = None

    out = _Eval.model_validate_json(raw)
    assert out.status == "completed"
    assert out.loss is not None
    assert out.ppl is not None


def test_eval_job_missing_manifest(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)

    def _fake_from_url(url: str, decode_responses: bool = False) -> fakeredis.FakeRedis:
        return fake

    monkeypatch.setattr("redis.from_url", _fake_from_url)

    artifacts = tmp_path / "artifacts"
    os.environ["APP__ARTIFACTS_ROOT"] = str(artifacts)
    _ = Settings()

    # Configure data-bank and pointer
    os.environ["APP__DATA_BANK_API_URL"] = "http://data-bank-api.local"
    os.environ["APP__DATA_BANK_API_KEY"] = "secret-key"

    run_id = "run-missing"
    file_id = "fid-missing"
    fake.set(f"runs:artifact:{run_id}:file_id", file_id)

    # Stub DataBankClient to return an archive without manifest.json
    from model_trainer.core.services.data import artifact_downloader as ad_mod

    tar_root = tmp_path / "db_missing"
    tar_root.mkdir()
    tar_path = tar_root / f"{run_id}.tar"
    name = f"model-{run_id}"
    # Create an archive with only weights.bin and no manifest.json
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"x")
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(model_dir):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(model_dir)
                arcname = Path(name) / rel
                tf.add(str(abs_path), arcname=str(arcname))

    class _ClientStub:
        def __init__(
            self: _ClientStub, base_url: str, api_key: str, timeout_seconds: float = 0.0
        ) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self: _ClientStub,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            assert file_id == "fid-missing"
            dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(ad_mod, "DataBankClient", _ClientStub)

    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    with pytest.raises(FileNotFoundError):
        process_eval_job(payload)

    raw = fake.get(f"{EVAL_KEY_PREFIX}{run_id}")
    assert raw is not None
    from pydantic import BaseModel

    class _Eval(BaseModel):
        status: str
        split: str

    out = _Eval.model_validate_json(raw)
    assert out.status == "failed"
