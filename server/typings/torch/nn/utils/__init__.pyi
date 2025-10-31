from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

def clip_grad_norm_(parameters: Iterable[Tensor], max_norm: float) -> None: ...
