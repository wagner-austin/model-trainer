from __future__ import annotations

import io
import tomllib
from pathlib import Path

import pytest
from model_trainer.core.config import settings as settings_mod
from model_trainer.core.config.settings import Settings
from pydantic_settings import PydanticBaseSettingsSource


class _DummySource(PydanticBaseSettingsSource):
    def __init__(self: _DummySource) -> None:
        super().__init__(Settings)

    def __call__(self: _DummySource) -> dict[str, object]:
        return {}

    def get_field_value(
        self: _DummySource, field: object, field_name: str
    ) -> tuple[object, str, bool]:
        return None, field_name, False


def test_toml_source_get_field_value_present_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make TOML source return a dict with one setting
    monkeypatch.setattr(settings_mod, "_load_toml_settings", lambda: {"app_env": "prod"})

    # Build sources and extract TOML source (index 2 in precedence tuple)
    sources = Settings.settings_customise_sources(
        Settings,
        _DummySource(),
        _DummySource(),
        _DummySource(),
        _DummySource(),
    )
    toml_source = sources[2]

    present = toml_source.get_field_value(field=None, field_name="app_env")
    assert present == ("prod", "app_env", True)

    missing = toml_source.get_field_value(field=None, field_name="does_not_exist")
    assert missing == (None, "does_not_exist", False)


def test_load_toml_non_dict_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Create a real file so os.path.exists() is True for the candidate path
    cfg = tmp_path / "app.toml"
    cfg.write_text("[x]\nkey='value'\n", encoding="utf-8")

    # Point loader to our file
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))

    # Force tomllib.load to return a non-dict to exercise the false branch at line 26
    def _fake_load(_: io.BufferedReader) -> str:  # return type intentionally non-dict
        return "not-a-dict"

    monkeypatch.setattr(tomllib, "load", _fake_load)

    out = settings_mod._load_toml_settings()
    # Because the first candidate returns a non-dict, the loop continues and we end with {}
    assert out == {}
