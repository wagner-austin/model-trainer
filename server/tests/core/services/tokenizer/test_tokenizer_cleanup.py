from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from model_trainer.core.config.settings import Settings, TokenizerCleanupConfig
from model_trainer.core.services.tokenizer.tokenizer_cleanup import (
    TokenizerCleanupError,
    TokenizerCleanupResult,
    TokenizerCleanupService,
    _directory_size,
)


def _settings_with_tokenizer_cleanup(cfg: TokenizerCleanupConfig, artifacts_root: Path) -> Settings:
    settings = Settings()
    settings.app.artifacts_root = str(artifacts_root)
    settings.app.tokenizer_cleanup = cfg
    return settings


def test_tokenizer_cleanup_disabled_returns_zero(tmp_path: Path) -> None:
    cfg = TokenizerCleanupConfig(enabled=False)
    settings = _settings_with_tokenizer_cleanup(cfg, tmp_path)
    svc = TokenizerCleanupService(settings=settings)

    result = svc.clean()

    assert result == TokenizerCleanupResult(deleted_tokenizers=0, bytes_freed=0)


def test_tokenizer_cleanup_missing_dir_is_noop(tmp_path: Path) -> None:
    cfg = TokenizerCleanupConfig(enabled=True)
    settings = _settings_with_tokenizer_cleanup(cfg, tmp_path)
    svc = TokenizerCleanupService(settings=settings)

    result = svc.clean()

    assert result.deleted_tokenizers == 0
    assert result.bytes_freed == 0


def test_tokenizer_referenced_in_manifest_is_not_deleted(tmp_path: Path) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    models_root = artifacts / "models"
    tokenizers_root.mkdir()
    models_root.mkdir()

    tok_dir = tokenizers_root / "tok1"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    run_dir = models_root / "run1"
    run_dir.mkdir()
    manifest = {
        "run_id": "run1",
        "tokenizer_id": "tok1",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=0)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    result = svc.clean()

    assert result.deleted_tokenizers == 0
    assert tok_dir.exists()


def test_unreferenced_old_tokenizer_is_deleted(tmp_path: Path) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    tokenizers_root.mkdir()

    tok_dir = tokenizers_root / "tok2"
    tok_dir.mkdir()
    tok_file = tok_dir / "tokenizer.json"
    tok_file.write_text("{}", encoding="utf-8")

    past = time.time() - 40 * 24 * 60 * 60
    os.utime(tok_dir, (past, past))
    os.utime(tok_file, (past, past))

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=30)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    result = svc.clean()

    assert result.deleted_tokenizers == 1
    assert result.bytes_freed > 0
    assert not tok_dir.exists()


def test_tokenizer_newer_than_threshold_not_deleted(tmp_path: Path) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    tokenizers_root.mkdir()
    tok_dir = tokenizers_root / "tok3"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=30)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    result = svc.clean()

    assert result.deleted_tokenizers == 0
    assert tok_dir.exists()


def test_tokenizer_cleanup_raises_on_non_directory_root(tmp_path: Path) -> None:
    root = tmp_path / "tokenizers"
    root.write_text("not a directory", encoding="utf-8")
    cfg = TokenizerCleanupConfig(enabled=True)
    settings = _settings_with_tokenizer_cleanup(cfg, tmp_path)
    svc = TokenizerCleanupService(settings=settings)

    with pytest.raises(TokenizerCleanupError):
        svc.clean()


def test_manifest_json_error_raises(tmp_path: Path) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    models_root = artifacts / "models"
    tokenizers_root.mkdir()
    models_root.mkdir()

    tok_dir = tokenizers_root / "tok4"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    run_dir = models_root / "run2"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{invalid json", encoding="utf-8")

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=0)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    with pytest.raises(TokenizerCleanupError):
        svc.clean()


def test_tokenizer_cleanup_deletion_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    tokenizers_root.mkdir()

    tok_dir = tokenizers_root / "tok_err"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    past = time.time() - 40 * 24 * 60 * 60
    os.utime(tok_dir, (past, past))

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=30)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    def _fail_rmtree(_path: Path) -> None:
        raise OSError("boom")

    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.tokenizer_cleanup.shutil.rmtree",
        _fail_rmtree,
    )

    with pytest.raises(TokenizerCleanupError):
        svc.clean()


def test_tokenizer_cleanup_oserror_on_iterdir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    tokenizers_root.mkdir()

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=0)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    def _iterdir_fail(_self: Path) -> None:
        raise OSError("iterdir-fail")

    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.tokenizer_cleanup.Path.iterdir",
        _iterdir_fail,
    )

    with pytest.raises(TokenizerCleanupError):
        svc.clean()


def test_collect_tokenizers_in_use_skips_non_dir_and_missing_manifest(tmp_path: Path) -> None:
    artifacts = tmp_path
    models_root = artifacts / "models"
    models_root.mkdir()
    # Non-directory entry under models root
    (models_root / "not_a_dir").write_text("x", encoding="utf-8")
    # Directory without manifest
    (models_root / "run_no_manifest").mkdir()

    cfg = TokenizerCleanupConfig(enabled=True)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    in_use = svc._collect_tokenizers_in_use()

    assert in_use == set()


def test_collect_tokenizers_in_use_ignores_blank_tokenizer_id(tmp_path: Path) -> None:
    artifacts = tmp_path
    models_root = artifacts / "models"
    models_root.mkdir()

    run_dir = models_root / "run_blank"
    run_dir.mkdir()
    manifest = {
        "run_id": "run_blank",
        "tokenizer_id": " ",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cfg = TokenizerCleanupConfig(enabled=True)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    in_use = svc._collect_tokenizers_in_use()

    assert in_use == set()


def test_directory_size_counts_nested_and_skips_dirs(tmp_path: Path) -> None:
    root = tmp_path / "tok_dir"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    f1 = root / "a.txt"
    f2 = nested / "b.txt"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("yz", encoding="utf-8")

    total = _directory_size(root)

    assert total == f1.stat().st_size + f2.stat().st_size


def test_tokenizer_cleanup_skips_non_directory_entries(tmp_path: Path) -> None:
    artifacts = tmp_path
    tokenizers_root = artifacts / "tokenizers"
    tokenizers_root.mkdir()
    # File that should be ignored by the cleanup loop
    marker = tokenizers_root / "marker.txt"
    marker.write_text("x", encoding="utf-8")

    tok_dir = tokenizers_root / "tok_non_dir"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    past = time.time() - 40 * 24 * 60 * 60
    os.utime(tok_dir, (past, past))

    cfg = TokenizerCleanupConfig(enabled=True, min_unused_days=30)
    settings = _settings_with_tokenizer_cleanup(cfg, artifacts)
    svc = TokenizerCleanupService(settings=settings)

    result = svc.clean()

    assert result.deleted_tokenizers == 1
    assert not tok_dir.exists()
    # Non-directory entry remains untouched
    assert marker.exists()
