from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest
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
        # Return a small, fixed sequence of ids for stable batching
        return _Enc([1, 2])

    def token_to_id(self: _Tok4DS, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _Tok4DS) -> int:
        return 4


class _LM:
    def __init__(self: _LM) -> None:
        self.saved_to: str | None = None

    def train(self: _LM) -> None:
        return None

    def eval(self: _LM) -> None:
        return None

    from model_trainer.core.services.model.backends.gpt2.types import ForwardOutProto

    def forward(self: _LM, *, input_ids: Tensor, labels: Tensor) -> ForwardOutProto:
        # Not used in this test (iterator yields no batches)
        raise NotImplementedError

    def parameters(self: _LM) -> Iterable[Tensor]:
        # Not used (optimizer monkeypatched to no-op)
        return []

    def to(self: _LM, device: object) -> object:
        return self

    def save_pretrained(self: _LM, out_dir: str) -> None:
        self.saved_to = out_dir

    @property
    def config(self: _LM) -> object:  # unused in training loop
        return {}


def test_train_prepared_gpt2_calls_progress_end_of_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Configure artifacts root
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    settings = Settings()

    # Create minimal corpus with one line
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello", encoding="utf-8")

    prepared = GPT2Prepared.__new__(GPT2Prepared)
    prepared.model = _LM()
    prepared.tokenizer_id = "tok"
    prepared.eos_id = 1
    prepared.pad_id = 0
    prepared.max_seq_len = 8
    prepared.tok_for_dataset = _Tok4DS()

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

    calls: list[tuple[int, int, float]] = []

    def _progress(step: int, epoch: int, loss: float) -> None:
        calls.append((step, epoch, loss))

    hb_times: list[float] = []

    def _hb(ts: float) -> None:
        hb_times.append(ts)

    # Replace DataLoader with an empty iterator to trigger end-of-epoch progress
    class _EmptyDL:
        def __init__(self: _EmptyDL, *args: object, **kwargs: object) -> None:
            pass

        def __iter__(self: _EmptyDL) -> object:
            return iter(())

    monkeypatch.setattr(gtrain, "DataLoader", _EmptyDL, raising=True)

    # Patch optimizer to avoid parameter list checks
    class _Opt:
        def __init__(self: _Opt, _params: Iterable[Tensor], lr: float) -> None:
            self._lr = lr

        def zero_grad(self: _Opt, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt) -> None:
            return None

    monkeypatch.setattr(gtrain, "AdamW", _Opt, raising=True)

    out = gtrain.train_prepared_gpt2(
        prepared,
        cfg,
        settings,
        run_id="r-prog",
        redis_hb=_hb,
        cancelled=lambda: False,
        progress=_progress,
    )

    # Expect single progress call at end of epoch (line 145)
    assert len(calls) == 1
    assert calls[-1][0] == 0 and calls[-1][1] == 0
    # Ensure outputs are sensible and artifacts were written
    assert out.steps == 0 and isinstance(out.loss, float)
    assert Path(out.out_dir).exists()
