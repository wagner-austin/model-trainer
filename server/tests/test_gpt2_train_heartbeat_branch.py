from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
import torch
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.model.backends.gpt2 import train as gtrain
from model_trainer.core.services.model.backends.gpt2.config import GPT2TrainConfig
from model_trainer.core.services.model.backends.gpt2.types import GPT2Prepared
from torch import Tensor


class _Enc:
    def __init__(self: _Enc, ids: list[int]) -> None:
        self._ids = ids

    @property
    def ids(self: _Enc) -> list[int]:
        return self._ids


class _Tok4DS:
    def encode(self: _Tok4DS, text: str) -> _Enc:
        return _Enc([1, 2])

    def token_to_id(self: _Tok4DS, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _Tok4DS) -> int:
        return 4


class _Out:
    @property
    def loss(self: _Out) -> Tensor:
        # Simple scalar requiring grad for backward()
        return torch.tensor([0.0], requires_grad=True)[0]


class _LM:
    def train(self: _LM) -> None:
        return None

    def eval(self: _LM) -> None:
        return None

    def forward(self: _LM, *, input_ids: Tensor, labels: Tensor) -> _Out:
        return _Out()

    def parameters(self: _LM) -> Iterable[Tensor]:
        return []

    def to(self: _LM, device: object) -> object:
        return self

    def save_pretrained(self: _LM, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    @property
    def config(self: _LM) -> object:  # unused
        return {}


class _Opt:
    def __init__(self: _Opt, _params: Iterable[Tensor], lr: float) -> None:
        self._lr = lr

    def zero_grad(self: _Opt, *, set_to_none: bool = True) -> None:
        return None

    def step(self: _Opt) -> None:
        return None


class _DL:
    def __init__(self: _DL, dataset: object, *, batch_size: int, shuffle: bool) -> None:
        # ignore dataset for speed; feed 10 batches to trigger step % 10 == 0
        self._n = 10

    def __iter__(self: _DL) -> Iterator[Tensor]:
        for _ in range(self._n):
            yield torch.tensor([1, 1], dtype=torch.long)


def test_train_prepared_gpt2_calls_heartbeat_every_10_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    settings = Settings()

    # Minimal corpus (dataset is ignored by patched DataLoader, but split requires one file)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello", encoding="utf-8")

    prepared = GPT2Prepared(
        model=_LM(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_Tok4DS(),
    )

    cfg = GPT2TrainConfig(
        model_family="gpt2",
        model_size="s",
        max_seq_len=8,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        tokenizer_id="tok",
        corpus_path=str(corpus),
    )

    # Patch fast optimizer and small DataLoader that yields 10 batches
    monkeypatch.setattr(gtrain, "AdamW", _Opt, raising=True)
    monkeypatch.setattr(gtrain, "DataLoader", _DL, raising=True)

    hb: list[float] = []
    progress_calls: list[tuple[int, int, float]] = []

    def _hb(ts: float) -> None:
        hb.append(ts)

    def _progress(step: int, epoch: int, loss: float) -> None:
        progress_calls.append((step, epoch, loss))

    out = gtrain.train_prepared_gpt2(
        prepared,
        cfg,
        settings,
        run_id="r-hb",
        redis_hb=_hb,
        cancelled=lambda: False,
        progress=_progress,
    )

    # Expect exactly one heartbeat at step == 10 (line 141)
    assert len(hb) == 1
    # Progress callback should be invoked once per batch in inner loop.
    assert len(progress_calls) == out.steps
    assert out.steps == 10
