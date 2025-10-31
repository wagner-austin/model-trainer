from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ....infra.persistence.models import (
    TrainingManifest,
    TrainingManifestVersions,
)
from ...config.settings import Settings
from ...contracts.dataset import DatasetBuilder, DatasetConfig
from .dataset_builder import CausalLMDataset, _TokenizerProto


def _gather_lib_versions() -> TrainingManifestVersions:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    def _v(name: str) -> str:
        try:
            return _pkg_version(name)
        except PackageNotFoundError as e:  # pragma: no cover - optional dependency
            logger = logging.getLogger(__name__)
            logger.warning("%s not available for version detection: %s", name, e)
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


class GPT2TrainConfig(BaseModel):
    model_family: Literal["gpt2"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    tokenizer_id: str
    corpus_path: str

    model_config = {"extra": "forbid", "validate_assignment": True}


"""Model I/O is typed via local stubs; keep training loop implementation generic."""


def _load_tokenizer(tokenizer_path: str) -> _TokenizerProto:
    from tokenizers import Tokenizer  # typed via local stubs

    return Tokenizer.from_file(tokenizer_path)

    # Dataset moved to dataset_builder


class TrainResult(BaseModel):
    loss: float
    perplexity: float
    steps: int
    out_dir: str
    cancelled: bool = False

    model_config = {"extra": "forbid", "validate_assignment": True}


def train_gpt2(
    cfg: GPT2TrainConfig,
    settings: Settings,
    *,
    run_id: str,
    redis_hb: Callable[[float], None],
    cancelled: Callable[[], bool],
    dataset_builder: DatasetBuilder,
) -> TrainResult:
    from transformers import GPT2Config, GPT2LMHeadModel  # typed via local stubs

    torch.manual_seed(42)
    random.seed(42)

    artifacts_root = settings.app.artifacts_root
    tokenizer_path = str(Path(artifacts_root) / "tokenizers" / cfg.tokenizer_id / "tokenizer.json")
    tokenizer = _load_tokenizer(tokenizer_path)
    eos_id_opt = tokenizer.token_to_id("[EOS]")
    eos_id = int(eos_id_opt) if eos_id_opt is not None else 0
    pad_id_opt = tokenizer.token_to_id("[PAD]")
    pad_id = int(pad_id_opt) if pad_id_opt is not None else 0
    vocab_size = int(tokenizer.get_vocab_size())

    ds_cfg = DatasetConfig(corpus_path=cfg.corpus_path, holdout_fraction=0.05)
    train_files, _ = dataset_builder.split(ds_cfg)

    dataset = CausalLMDataset(
        files=train_files,
        tokenizer=tokenizer,
        max_len=cfg.max_seq_len,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    dataloader: DataLoader[Tensor] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Tiny GPT-2 config suitable for CPU
    gpt2_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.max_seq_len,
        n_ctx=cfg.max_seq_len,
        n_embd=128,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    model.train()
    device = torch.device("cpu")
    model.to(device)

    optim = AdamW(model.parameters(), lr=cfg.learning_rate)

    step = 0
    last_loss = 0.0
    was_cancelled = False

    def _iter_batches(loader: Iterable[Tensor]) -> Iterator[Tensor]:
        yield from loader

    for _epoch in range(cfg.num_epochs):
        for batch in _iter_batches(dataloader):
            if cancelled():
                was_cancelled = True
                break
            batch_t: Tensor = batch
            inputs: Tensor = batch_t.to(device)
            outputs = model.forward(input_ids=inputs, labels=inputs)  # loss inside
            loss_t: Tensor = outputs.loss
            last_loss = float(loss_t.item())
            optim.zero_grad(set_to_none=True)
            loss_t.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            step += 1
            if step % 10 == 0:
                redis_hb(time.time())
        if was_cancelled:
            break

    out_dir = str(Path(artifacts_root) / "models" / run_id)
    os.makedirs(out_dir, exist_ok=True)
    # Save model weights only if not cancelled
    if not was_cancelled:
        model.save_pretrained(out_dir)
    # Save training manifest
    # library versions
    import platform as _platform

    vers: TrainingManifestVersions = _gather_lib_versions()

    manifest: TrainingManifest = {
        "run_id": run_id,
        "epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "max_seq_len": cfg.max_seq_len,
        "steps": step,
        "loss": last_loss,
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
        loss=last_loss, perplexity=ppl, steps=step, out_dir=out_dir, cancelled=was_cancelled
    )


class EvalResult(BaseModel):
    loss: float
    perplexity: float

    model_config = {"extra": "forbid", "validate_assignment": True}


def evaluate_gpt2(
    *, run_id: str, cfg: GPT2TrainConfig, settings: Settings, dataset_builder: DatasetBuilder
) -> EvalResult:
    from transformers import GPT2LMHeadModel

    artifacts_root = settings.app.artifacts_root
    tokenizer_path = str(Path(artifacts_root) / "tokenizers" / cfg.tokenizer_id / "tokenizer.json")
    tokenizer = _load_tokenizer(tokenizer_path)
    eos_id_opt = tokenizer.token_to_id("[EOS]")
    eos_id = int(eos_id_opt) if eos_id_opt is not None else 0
    pad_id_opt = tokenizer.token_to_id("[PAD]")
    pad_id = int(pad_id_opt) if pad_id_opt is not None else 0

    ds_cfg = DatasetConfig(corpus_path=cfg.corpus_path, holdout_fraction=0.05)
    _, val_files = dataset_builder.split(ds_cfg)
    dataset = CausalLMDataset(
        files=val_files, tokenizer=tokenizer, max_len=cfg.max_seq_len, eos_id=eos_id, pad_id=pad_id
    )
    dataloader: DataLoader[Tensor] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model_dir = str(Path(artifacts_root) / "models" / run_id)
    # Load model via typed API; avoids torch.load
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    total_loss = 0.0
    total_count = 0
    eval_dir = Path(settings.app.artifacts_root) / "models" / run_id / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_t: Tensor = batch
            inputs: Tensor = batch_t.to(device)
            outputs = model.forward(input_ids=inputs, labels=inputs)
            loss_t: Tensor = outputs.loss
            batch_count: int = int(inputs.size(0))
            total_loss += float(loss_t.item()) * float(batch_count)
            total_count += batch_count
    avg_loss = total_loss / max(1, total_count)
    ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
    # Persist metrics
    metrics = {"loss": avg_loss, "perplexity": ppl}
    with open(str(eval_dir / "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))
    return EvalResult(loss=avg_loss, perplexity=ppl)
