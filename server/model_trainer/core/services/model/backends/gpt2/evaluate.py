from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetBuilder, DatasetConfig
from model_trainer.core.infra.paths import model_dir as _model_dir
from model_trainer.core.infra.paths import model_eval_dir
from model_trainer.core.services.training.dataset_builder import CausalLMDataset

from .config import GPT2TrainConfig
from .io import load_tokenizer_for_dataset, token_ids


class EvalResult(BaseModel):
    loss: float
    perplexity: float

    model_config = {"extra": "forbid", "validate_assignment": True}


def evaluate_gpt2(
    *, run_id: str, cfg: GPT2TrainConfig, settings: Settings, dataset_builder: DatasetBuilder
) -> EvalResult:
    from transformers import GPT2LMHeadModel  # typed via local stubs

    artifacts_root = settings.app.artifacts_root
    tokenizer_path = str(Path(artifacts_root) / "tokenizers" / cfg.tokenizer_id / "tokenizer.json")
    tokenizer = load_tokenizer_for_dataset(tokenizer_path)
    eos_id, pad_id, _ = token_ids(tokenizer)

    ds_cfg = DatasetConfig(corpus_path=cfg.corpus_path, holdout_fraction=0.05)
    _, val_files = dataset_builder.split(ds_cfg)
    dataset = CausalLMDataset(
        files=val_files, tokenizer=tokenizer, max_len=cfg.max_seq_len, eos_id=eos_id, pad_id=pad_id
    )
    dataloader: DataLoader[Tensor] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model_dir = str(_model_dir(settings, run_id))
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    total_loss = 0.0
    total_count = 0
    eval_dir = model_eval_dir(settings, run_id)
    eval_dir.mkdir(parents=True, exist_ok=True)
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
    metrics = {"loss": avg_loss, "perplexity": ppl}
    with open(str(eval_dir / "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))
    return EvalResult(loss=avg_loss, perplexity=ppl)
