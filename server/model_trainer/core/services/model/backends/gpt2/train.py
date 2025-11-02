from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.infra.paths import model_dir
from model_trainer.core.services.training.dataset_builder import (
    CausalLMDataset,
    split_corpus_files,
)
from model_trainer.infra.persistence.models import TrainingManifest, TrainingManifestVersions

from .config import GPT2TrainConfig
from .types import GPT2Prepared


def _gather_lib_versions() -> TrainingManifestVersions:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    def _v(name: str) -> str:
        try:
            return _pkg_version(name)
        except PackageNotFoundError as e:  # pragma: no cover - optional dependency
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "%s not available for version detection: %s", name, e
            )
            return "unknown"

    return {
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "tokenizers": _v("tokenizers"),
        "datasets": _v("datasets"),
    }


def _maybe_git_commit(settings: Settings) -> str | None:
    try:
        import subprocess as _sp

        repo_root = Path(settings.app.artifacts_root).parents[1]
        return (
            _sp.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=_sp.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (_sp.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover - optional
        return None


class TrainResult(BaseModel):
    loss: float
    perplexity: float
    steps: int
    out_dir: str
    cancelled: bool = False

    model_config = {"extra": "forbid", "validate_assignment": True}


def train_prepared_gpt2(
    prepared: GPT2Prepared,
    cfg: GPT2TrainConfig,
    settings: Settings,
    *,
    run_id: str,
    redis_hb: Callable[[float], None],
    cancelled: Callable[[], bool],
    progress: Callable[[int, int, float], None] | None = None,
) -> TrainResult:
    import platform as _platform

    torch.manual_seed(42)
    random.seed(42)

    eos_id = prepared.eos_id
    pad_id = prepared.pad_id

    # Simple split assuming on-disk data
    ds_cfg = DatasetConfig(corpus_path=cfg.corpus_path, holdout_fraction=0.05)
    train_files, _ = split_corpus_files(ds_cfg)

    dataset = CausalLMDataset(
        files=train_files,
        tokenizer=prepared.tok_for_dataset,
        max_len=prepared.max_seq_len,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    dataloader: DataLoader[Tensor] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = prepared.model
    model.train()
    device = torch.device("cpu")
    model.to(device)

    # Optimizer
    from torch.optim import AdamW

    optim = AdamW(model.parameters(), lr=cfg.learning_rate)

    step = 0
    last_loss = 0.0
    was_cancelled = False

    def _iter_batches(loader: Iterable[Tensor]) -> Iterator[Tensor]:
        yield from loader

    for epoch in range(cfg.num_epochs):
        for batch in _iter_batches(dataloader):
            if cancelled():
                was_cancelled = True
                break
            inputs: Tensor = batch.to(device)
            outputs = model.forward(input_ids=inputs, labels=inputs)
            loss_t: Tensor = outputs.loss
            last_loss = float(loss_t.item())
            optim.zero_grad(set_to_none=True)
            loss_t.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            step += 1
            if progress is not None:
                progress(step, epoch, last_loss)
            if step % 10 == 0:
                redis_hb(time.time())
        if was_cancelled:
            break
        if progress is not None:
            progress(step, epoch, last_loss)

    out_dir = str(model_dir(settings, run_id))
    os.makedirs(out_dir, exist_ok=True)
    # Save model weights unless cancelled (so evaluation can load from disk)
    if not was_cancelled:
        prepared.model.save_pretrained(out_dir)
    # Persist training manifest (weights saved separately by save lifecycle)
    vers: TrainingManifestVersions = _gather_lib_versions()
    manifest: TrainingManifest = {
        "run_id": run_id,
        "model_family": cfg.model_family,
        "model_size": cfg.model_size,
        "epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "max_seq_len": cfg.max_seq_len,
        "steps": step,
        "loss": last_loss,
        "learning_rate": cfg.learning_rate,
        "tokenizer_id": cfg.tokenizer_id,
        "corpus_path": cfg.corpus_path,
        "optimizer": "AdamW",
        "seed": 42,
        "versions": vers,
        "system": {
            "cpu_count": int(os.cpu_count() or 1),
            "platform": _platform.system(),
            "platform_release": _platform.release(),
            "machine": _platform.machine(),
        },
        "git_commit": _maybe_git_commit(settings),
    }
    with open(str(Path(out_dir) / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))

    ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")
    return TrainResult(
        loss=last_loss,
        perplexity=ppl,
        steps=step,
        out_dir=out_dir,
        cancelled=was_cancelled,
    )
