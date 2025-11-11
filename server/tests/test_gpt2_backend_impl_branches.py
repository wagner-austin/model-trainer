from __future__ import annotations

from types import SimpleNamespace

import pytest
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.model.gpt2_backend_impl import GPT2BackendImpl


def test_gpt2_backend_load_calls_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_load(path: str, tok: object) -> object:
        called["path"] = path
        called["tok"] = tok
        return SimpleNamespace(name="ok")

    monkeypatch.setattr(
        "model_trainer.core.services.model.gpt2_backend_impl.load_prepared_gpt2_from_handle",
        _fake_load,
    )

    class _Tok(TokenizerHandle):
        def encode(self: _Tok, text: str) -> list[int]:
            return [1]

        def decode(self: _Tok, ids: list[int]) -> str:
            return "x"

        def token_to_id(self: _Tok, token: str) -> int | None:
            return 0

        def get_vocab_size(self: _Tok) -> int:
            return 1

    ds = SimpleNamespace()
    backend = GPT2BackendImpl(ds)  # dataset builder unused here
    _ = backend.load("/x", Settings(), tokenizer=_Tok())
    # Function was invoked; validate captured path as proxy for behavior
    assert called["path"] == "/x"


def test_gpt2_backend_train_type_error() -> None:
    ds = SimpleNamespace()
    backend = GPT2BackendImpl(ds)

    class _BadPrepared:
        pass

    bad_prepared: PreparedModel = _BadPrepared()
    with pytest.raises(TypeError):
        _ = backend.train(
            ModelTrainConfig(
                model_family="gpt2",
                model_size="small",
                max_seq_len=8,
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-4,
                tokenizer_id="t",
                corpus_path="/c",
            ),
            Settings(),
            run_id="r",
            heartbeat=lambda _: None,
            cancelled=lambda: False,
            prepared=bad_prepared,
        )
