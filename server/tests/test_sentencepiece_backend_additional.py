from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.tokenizer.spm_backend import (
    SentencePieceBackend,
)


def test_spm_encode_ids_blank_output_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Prepare a dummy model path (not used by the stubbed subprocess)
    model_path = tmp_path / "tok.model"
    model_path.write_bytes(b"")

    # Stub subprocess.run to emit only blank lines for spm_encode
    class _Proc:
        def __init__(self: _Proc, stdout: str) -> None:
            self.stdout = stdout

    def _run(args: list[str] | tuple[str, ...], *a: object, **k: object) -> _Proc:
        # Only handle encode here; other commands return empty output
        cmd = str(args[0])
        if "spm_encode" in cmd:
            return _Proc("\n   \n\n")  # all blank/whitespace lines
        return _Proc("")

    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.spm_backend.subprocess.run", _run, raising=True
    )

    # Call encode through adapter and assert empty ids
    h = SentencePieceBackend().load(str(model_path))
    ids = SentencePieceBackend().encode(h, "ignored")
    assert ids == []


def test_train_spm_tokenizer_no_files_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Disable CLI check and call train with an empty corpus dir
    import model_trainer.core.services.tokenizer.spm_backend as spm

    monkeypatch.setattr(spm, "_require_cli", lambda: None, raising=True)

    backend = SentencePieceBackend()
    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(tmp_path / "empty"),
        holdout_fraction=0.2,
        seed=1,
        out_dir=str(tmp_path / "out"),
    )
    with pytest.raises(RuntimeError):
        _ = backend.train(cfg)


def test_train_spm_tokenizer_clamps_sample_and_char_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import model_trainer.core.services.tokenizer.spm_backend as spm

    # Disable real CLI and training; write minimal model/vocab
    monkeypatch.setattr(spm, "_require_cli", lambda: None, raising=True)

    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        Path(model_prefix + ".model").write_bytes(b"m")
        Path(model_prefix + ".vocab").write_text("[UNK]\nA\nB\n", encoding="utf-8")

    monkeypatch.setattr(spm, "_spm_train", _fake_train, raising=True)

    # Encode stub: return [0] for 'x' (unknown), [1] otherwise (known)
    def _fake_encode(_model: str, text: str) -> list[int]:
        if len(text) == 1 and text.lower() == "x":
            return [0]
        return [1]

    monkeypatch.setattr(spm, "_spm_encode_ids", _fake_encode, raising=True)

    # Build corpus with enough lines to trigger holdout clamp when sample_max_lines=1
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = ["xy", "ab", "cd", "ef", "gh", "ij", "kl", "mn", "op", "qr"]
    (corpus / "a.txt").write_text("\n".join(lines), encoding="utf-8")

    out_dir = tmp_path / "tok"
    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=32,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=42,
        out_dir=str(out_dir),
        sample_max_lines=1,  # clamp to 1
    )
    stats = SentencePieceBackend().train(cfg)
    # Ensure manifest written and stats within expected bounds
    assert (out_dir / "manifest.json").exists()
    assert 0.0 <= stats.coverage <= 1.0
    assert 0.0 <= stats.char_coverage <= 1.0


def test_train_spm_tokenizer_no_clamp_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import model_trainer.core.services.tokenizer.spm_backend as spm

    # Avoid real CLI
    monkeypatch.setattr(spm, "_require_cli", lambda: None, raising=True)

    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        Path(model_prefix + ".model").write_bytes(b"m")
        Path(model_prefix + ".vocab").write_text("[UNK]\nA\nB\n", encoding="utf-8")

    monkeypatch.setattr(spm, "_spm_train", _fake_train, raising=True)

    # Encode returns a single non-UNK id to exercise the true branch of coverage calc
    def _enc_ids(_m: str, _t: str) -> list[int]:
        return [1]

    monkeypatch.setattr(spm, "_spm_encode_ids", _enc_ids, raising=True)

    corpus = tmp_path / "corpus_no_clamp"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abc\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=16,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=7,
        out_dir=str(tmp_path / "tok_nc"),
        # sample_max_lines left as None to exercise false path of clamp
    )
    stats = SentencePieceBackend().train(cfg)
    assert stats.token_count >= 0


def test_train_spm_tokenizer_empty_ids_skip_in_char_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import model_trainer.core.services.tokenizer.spm_backend as spm

    monkeypatch.setattr(spm, "_require_cli", lambda: None, raising=True)

    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        Path(model_prefix + ".model").write_bytes(b"m")
        Path(model_prefix + ".vocab").write_text("[UNK]\nA\n", encoding="utf-8")

    monkeypatch.setattr(spm, "_spm_train", _fake_train, raising=True)

    # For a specific char, return empty ids to trigger the false arm from the ids-empty side
    def _enc_ids_empty(_m: str, _t: str) -> list[int]:
        return []

    monkeypatch.setattr(spm, "_spm_encode_ids", _enc_ids_empty, raising=True)

    corpus = tmp_path / "corpus_empty_ids"
    corpus.mkdir()
    (corpus / "a.txt").write_text("x\n", encoding="utf-8")

    cfg = TokenizerTrainConfig(
        method="sentencepiece",
        vocab_size=8,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.5,
        seed=3,
        out_dir=str(tmp_path / "tok_ei"),
    )
    stats = SentencePieceBackend().train(cfg)
    assert stats.char_coverage in (0.0, 1.0)


def test_spm_inspect_success_reads_manifest(tmp_path: Path) -> None:
    # Create a directory with a valid SentencePiece-style manifest
    out = tmp_path / "tok"
    out.mkdir()
    manifest = {
        "stats": {
            "coverage": 0.75,
            "oov_rate": 0.1,
            "token_count": 123,
            "char_coverage": 0.8,
        }
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stats = SentencePieceBackend().inspect(str(out))
    assert stats.token_count == 123 and stats.oov_rate == 0.1
