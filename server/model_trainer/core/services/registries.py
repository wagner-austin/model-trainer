from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..contracts.model import ModelBackend
from ..contracts.tokenizer import TokenizerBackend


@dataclass
class ModelRegistry:
    backends: Mapping[str, ModelBackend]

    def get(self: ModelRegistry, name: str) -> ModelBackend:
        if name not in self.backends:
            raise KeyError(f"Unknown model backend: {name}")
        return self.backends[name]


@dataclass
class TokenizerRegistry:
    backends: Mapping[str, TokenizerBackend]

    def get(self: TokenizerRegistry, method: str) -> TokenizerBackend:
        if method not in self.backends:
            raise KeyError(f"Unknown tokenizer backend: {method}")
        return self.backends[method]
