from __future__ import annotations

from pydantic import BaseModel

class PydanticBaseSettingsSource:
    def __init__(self: PydanticBaseSettingsSource, settings_cls: type[BaseSettings]) -> None: ...
    def __call__(self: PydanticBaseSettingsSource) -> dict[str, object]: ...
    def get_field_value(
        self: PydanticBaseSettingsSource, field: object, field_name: str
    ) -> tuple[object, str, bool]: ...

class BaseSettings(BaseModel):
    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]: ...
