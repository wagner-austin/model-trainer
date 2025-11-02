from __future__ import annotations

import os
from pathlib import Path

from model_trainer.core.config.settings import Settings
from pytest import MonkeyPatch


def test_settings_loads_from_toml_and_env_override(tmp_path: Path) -> None:
    cfg = tmp_path / "app.toml"
    cfg.write_text(
        """
[app]
artifacts_root = "/x/artifacts"
runs_root = "/x/runs"
""".strip(),
        encoding="utf-8",
    )
    os.environ["APP_CONFIG_FILE"] = str(cfg)
    # Override via env for precedence
    os.environ["APP__ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    s = Settings()
    assert s.app.artifacts_root == str(tmp_path / "artifacts")
    assert s.app.runs_root == "/x/runs"


def test_settings_toml_missing_or_unreadable(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Point to non-existent file: should not crash
    os.environ["APP_CONFIG_FILE"] = str(tmp_path / "nope.toml")
    _ = Settings()
    # Simulate OSError on open
    import builtins as _b

    def _bad_open(
        file: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: object | None = None,
    ) -> object:  # return type unused; function raises
        raise OSError("boom")

    monkeypatch.setattr(_b, "open", _bad_open)
    _ = Settings()
