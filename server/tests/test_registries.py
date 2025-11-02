from __future__ import annotations

import pytest
from model_trainer.core.services.model.unavailable_backend import UnavailableBackend
from model_trainer.core.services.registries import ModelRegistry, TokenizerRegistry


def test_model_registry_get_and_missing() -> None:
    reg = ModelRegistry(backends={"llama": UnavailableBackend("llama")})
    b = reg.get("llama")
    assert b.name() == "llama"
    with pytest.raises(KeyError):
        _ = reg.get("nope")


def test_tokenizer_registry_missing() -> None:
    reg = TokenizerRegistry(backends={})
    with pytest.raises(KeyError):
        _ = reg.get("nope")
