from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from pydantic import BaseModel
from tokenizers import Tokenizer as _Tok  # typed via local stubs

from ...contracts.tokenizer import (
    TokenizerBackend as _TokenizerBackendProto,
)
from ...contracts.tokenizer import (
    TokenizerHandle as _TokenizerHandle,
)
from ...contracts.tokenizer import (
    TokenizerTrainConfig as _TokenizerTrainConfig,
)
from ...contracts.tokenizer import (
    TokenizerTrainStats as _TokenizerTrainStats,
)
from ..data.corpus import iter_lines, list_text_files


class TokenizerStats(BaseModel):
    coverage: float
    oov_rate: float
    token_count: int
    char_coverage: float

    model_config = {"extra": "forbid", "validate_assignment": True}


class BPETrainConfig(BaseModel):
    vocab_size: int
    min_frequency: int
    holdout_fraction: float
    seed: int
    special_tokens: tuple[str, ...] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")

    model_config = {"extra": "forbid", "validate_assignment": True}


def train_bpe_tokenizer(corpus_path: str, out_dir: str, cfg: BPETrainConfig) -> TokenizerStats:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    os.makedirs(out_dir, exist_ok=True)
    files = list_text_files(corpus_path)
    if not files:
        raise RuntimeError(f"No text files found under {corpus_path}")

    tokenizer = Tokenizer(BPE(unk_token=cfg.special_tokens[1]))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=list(cfg.special_tokens),
        show_progress=False,
    )
    tokenizer.train(files=files, trainer=trainer)
    tok_path = str(Path(out_dir) / "tokenizer.json")
    tokenizer.save(tok_path)

    # Validation stats on a small holdout sample
    lines = list(iter_lines(files))
    random.seed(cfg.seed)
    random.shuffle(lines)
    holdout_n = max(1, int(len(lines) * cfg.holdout_fraction))
    sample = lines[:holdout_n]
    total_tokens = 0
    unk_tokens = 0
    for s in sample:
        enc = tokenizer.encode(s)
        total_tokens += len(enc.ids)
        # OOV approximated as count of UNK id occurrences
        _unk = tokenizer.token_to_id(cfg.special_tokens[1])
        unk_id_to_count = _unk if _unk is not None else -1
        unk_tokens += int(enc.ids.count(unk_id_to_count))
    coverage = 1.0 if total_tokens == 0 else max(0.0, 1.0 - (unk_tokens / max(1, total_tokens)))

    # Character coverage: fraction of distinct characters that encode to any non-UNK id
    # This avoids relying on offsets and gives a stable [0,1] signal across corpora.
    uniq_chars = set("".join(sample))
    covered_chars = 0
    unk_id = tokenizer.token_to_id(cfg.special_tokens[1])
    for ch in uniq_chars:
        ids = tokenizer.encode(ch).ids
        if not ids:
            continue
        if unk_id is None or any(tid != unk_id for tid in ids):
            covered_chars += 1
    char_cov = 1.0 if len(uniq_chars) == 0 else max(0.0, min(1.0, covered_chars / len(uniq_chars)))
    stats = TokenizerStats(
        coverage=coverage,
        oov_rate=(unk_tokens / max(1, total_tokens)),
        token_count=total_tokens,
        char_coverage=char_cov,
    )
    # Persist manifest
    manifest: dict[str, object] = {
        "created_at": int(time.time()),
        "config": {
            "vocab_size": cfg.vocab_size,
            "min_frequency": cfg.min_frequency,
            "holdout_fraction": cfg.holdout_fraction,
            "seed": cfg.seed,
            "special_tokens": list(cfg.special_tokens),
        },
        "stats": {
            "coverage": float(stats.coverage),
            "oov_rate": float(stats.oov_rate),
            "token_count": int(stats.token_count),
            "char_coverage": float(stats.char_coverage),
        },
    }
    with open(str(Path(out_dir) / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))
    return stats


class _TokenizerAdapter:
    def __init__(self: _TokenizerAdapter, inner: _Tok) -> None:
        self._inner: _Tok = inner

    def encode(self: _TokenizerAdapter, text: str) -> list[int]:
        enc = self._inner.encode(text)
        return list(enc.ids)

    def decode(self: _TokenizerAdapter, ids: list[int]) -> str:
        return self._inner.decode(ids)

    def token_to_id(self: _TokenizerAdapter, token: str) -> int | None:
        return self._inner.token_to_id(token)

    def get_vocab_size(self: _TokenizerAdapter) -> int:
        return int(self._inner.get_vocab_size())


class BPEBackend(_TokenizerBackendProto):
    def name(self: BPEBackend) -> str:
        return "bpe"

    def train(self: BPEBackend, cfg: _TokenizerTrainConfig) -> _TokenizerTrainStats:
        stats = train_bpe_tokenizer(
            corpus_path=cfg.corpus_path,
            out_dir=cfg.out_dir,
            cfg=BPETrainConfig(
                vocab_size=cfg.vocab_size,
                min_frequency=cfg.min_frequency,
                holdout_fraction=cfg.holdout_fraction,
                seed=cfg.seed,
            ),
        )
        return _TokenizerTrainStats(
            coverage=stats.coverage,
            oov_rate=stats.oov_rate,
            token_count=stats.token_count,
            char_coverage=stats.char_coverage,
        )

    def load(self: BPEBackend, artifact_path: str) -> _TokenizerHandle:
        from tokenizers import Tokenizer  # typed via local stubs

        tok = Tokenizer.from_file(artifact_path)
        return _TokenizerAdapter(tok)

    def encode(self: BPEBackend, handle: _TokenizerHandle, text: str) -> list[int]:
        return handle.encode(text)

    def decode(self: BPEBackend, handle: _TokenizerHandle, ids: list[int]) -> str:
        return handle.decode(ids)
