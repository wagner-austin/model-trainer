from __future__ import annotations

from typing import Protocol

import torch
from torch.utils.data import Dataset

from ...contracts.dataset import DatasetConfig
from ..data.corpus import list_text_files


def split_corpus_files(cfg: DatasetConfig) -> tuple[list[str], list[str]]:
    files = list_text_files(cfg.corpus_path)
    if not files:
        raise RuntimeError(f"No text files found under {cfg.corpus_path}")
    n = len(files)
    val_n = max(1, int(n * cfg.holdout_fraction))
    train_files = files[:-val_n] if n > 1 else files
    val_files = files[-val_n:] if n > 1 else files
    return train_files, val_files


class _TokenizerProto(Protocol):
    def encode(self: _TokenizerProto, text: str) -> _EncodingProto: ...
    def token_to_id(self: _TokenizerProto, token: str) -> int | None: ...
    def get_vocab_size(self: _TokenizerProto) -> int: ...


class _EncodingProto(Protocol):
    @property
    def ids(self: _EncodingProto) -> list[int]: ...


class CausalLMDataset(Dataset[torch.Tensor]):
    def __init__(
        self: CausalLMDataset,
        *,
        files: list[str],
        tokenizer: _TokenizerProto,
        max_len: int,
        eos_id: int,
        pad_id: int,
    ) -> None:
        self._ids: list[int] = []
        for fp in files:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    enc = tokenizer.encode(s)
                    self._ids.extend([*enc.ids, eos_id])
        self._max_len = max_len
        self._pad_id = pad_id

    def __len__(self: CausalLMDataset) -> int:
        if not self._ids:
            return 0
        # Number of chunks, include partial trailing chunk
        return max(1, (len(self._ids) + self._max_len - 1) // self._max_len)

    def __getitem__(self: CausalLMDataset, idx: int) -> torch.Tensor:
        start = idx * self._max_len
        end = start + self._max_len
        chunk = self._ids[start:end]
        if len(chunk) < self._max_len:
            pad = [self._pad_id] * (self._max_len - len(chunk))
            chunk = chunk + pad
        return torch.tensor(chunk, dtype=torch.long)
