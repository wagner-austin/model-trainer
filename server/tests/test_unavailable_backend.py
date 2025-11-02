from __future__ import annotations

import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.model.unavailable_backend import UnavailableBackend


class _TokHandle:
    def encode(self: _TokHandle, text: str) -> list[int]:
        return [1]

    def decode(self: _TokHandle, ids: list[int]) -> str:
        return ""

    def token_to_id(self: _TokHandle, token: str) -> int | None:
        return None

    def get_vocab_size(self: _TokHandle) -> int:
        return 1


def test_unavailable_backend_all_methods_raise() -> None:
    ub = UnavailableBackend("llama")
    cfg = ModelTrainConfig(
        model_family="gpt2",
        model_size="s",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        tokenizer_id="tok",
        corpus_path="/c",
    )
    s = Settings()
    from model_trainer.core.errors.base import AppError

    with pytest.raises(AppError):
        _ = ub.prepare(cfg, s, tokenizer=_TokHandle())
    with pytest.raises(AppError):
        _ = ub.save(object(), "/out")
    with pytest.raises(AppError):
        _ = ub.load("/in", s, tokenizer=_TokHandle())
    with pytest.raises(AppError):
        _ = ub.train(
            cfg,
            s,
            run_id="r1",
            heartbeat=lambda t: None,
            cancelled=lambda: False,
            prepared=object(),
        )
    with pytest.raises(AppError):
        _ = ub.evaluate(run_id="r1", cfg=cfg, settings=s)
