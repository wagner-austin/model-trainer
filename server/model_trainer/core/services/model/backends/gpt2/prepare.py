from __future__ import annotations

from pathlib import Path

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.training.dataset_builder import _TokenizerProto

from .config import GPT2TrainConfig
from .io import _TokWrapper, load_tokenizer_for_dataset, token_ids
from .types import GPT2Prepared, TokenHandleProto


def prepare_gpt2(cfg: GPT2TrainConfig, settings: Settings) -> GPT2Prepared:
    from transformers import GPT2Config, GPT2LMHeadModel  # typed via local stubs

    artifacts_root = settings.app.artifacts_root
    tokenizer_path = str(Path(artifacts_root) / "tokenizers" / cfg.tokenizer_id / "tokenizer.json")
    tokenizer: _TokenizerProto = load_tokenizer_for_dataset(tokenizer_path)
    eos_id, pad_id, vocab_size = token_ids(tokenizer)
    gpt2_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.max_seq_len,
        n_ctx=cfg.max_seq_len,
        n_embd=128,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    return GPT2Prepared(
        model=model,
        tokenizer_id=cfg.tokenizer_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg.max_seq_len,
        tok_for_dataset=tokenizer,
    )


def prepare_gpt2_with_handle(
    tokenizer: TokenizerHandle | TokenHandleProto, cfg: GPT2TrainConfig
) -> GPT2Prepared:
    from transformers import GPT2Config, GPT2LMHeadModel  # typed via local stubs

    eos_id, pad_id, vocab_size = token_ids(tokenizer)
    gpt2_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.max_seq_len,
        n_ctx=cfg.max_seq_len,
        n_embd=128,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    return GPT2Prepared(
        model=model,
        tokenizer_id=cfg.tokenizer_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg.max_seq_len,
        tok_for_dataset=_TokWrapper(tokenizer),
    )
