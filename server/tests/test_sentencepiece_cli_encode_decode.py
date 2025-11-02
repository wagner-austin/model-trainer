from __future__ import annotations

from pathlib import Path

from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend
from pytest import MonkeyPatch


class _CP:
    def __init__(self: _CP, stdout: str) -> None:
        self.stdout = stdout


def test_sentencepiece_adapter_encode_decode(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Provide model + vocab files
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"")
    (tmp_path / "tokenizer.vocab").write_text("[PAD]\n[UNK]\n[EOS]\nfoo\nbar\n", encoding="utf-8")

    # Pretend CLI exists and returns fixed outputs
    import shutil as _shutil
    import subprocess as _sp

    def _which(_: str) -> str:
        return "/bin/true"

    def _run(args: list[str] | tuple[str, ...], *a: object, **k: object) -> _CP:
        if args and "spm_encode" in args[0]:
            return _CP("1 2 3\n")
        if args and "spm_decode" in args[0]:
            return _CP("hello\n")
        return _CP("")

    monkeypatch.setattr(_shutil, "which", _which)
    monkeypatch.setattr(_sp, "run", _run)

    handle = SentencePieceBackend().load(str(model_path))
    ids = SentencePieceBackend().encode(handle, "foo bar")
    text = SentencePieceBackend().decode(handle, ids)
    assert ids == [1, 2, 3]
    assert text == "hello"
