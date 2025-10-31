from __future__ import annotations

import logging
import os
import tomllib
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


def _load_toml_settings() -> dict[str, object]:
    # Load from a TOML config if available.
    # Search: APP_CONFIG_FILE env, then ./server/config/app.toml, then ./config/app.toml
    candidates: list[str] = []
    env_path = os.getenv("APP_CONFIG_FILE")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), "server", "config", "app.toml"))
    candidates.append(os.path.join(os.getcwd(), "config", "app.toml"))
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    raw = tomllib.load(f)
                if isinstance(raw, dict):
                    out: dict[str, object] = {}
                    for k, v in raw.items():
                        out[k] = v
                    return out
        except OSError:
            logging.getLogger(__name__).warning("Failed to read config file %s", p)
            return {}
    return {}


class LoggingConfig(BaseModel):
    level: str = "INFO"

    model_config = {"extra": "forbid", "validate_assignment": True}


class RedisConfig(BaseSettings):
    enabled: bool = True
    url: str = "redis://redis:6379/0"

    model_config = {
        "extra": "forbid",
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }


class RQConfig(BaseSettings):
    queue_name: str = "training"
    job_timeout_sec: int = 86_400  # 24h
    result_ttl_sec: int = 86_400
    failure_ttl_sec: int = 7 * 86_400
    retry_max: int = 1
    retry_intervals_sec: str = "300"  # comma-separated

    model_config = {
        "extra": "forbid",
        "env_nested_delimiter": "__",
    }


class AppConfig(BaseSettings):
    data_root: str = "/data"
    artifacts_root: str = "/data/artifacts"
    runs_root: str = "/data/runs"
    logs_root: str = "/data/logs"

    model_config = {
        "extra": "forbid",
        "env_nested_delimiter": "__",
    }


class Settings(BaseSettings):
    app_env: Literal["dev", "prod"] = "dev"
    logging: LoggingConfig = LoggingConfig()
    redis: RedisConfig = RedisConfig()
    rq: RQConfig = RQConfig()
    app: AppConfig = AppConfig()

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "forbid",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def settings_customise_sources(
        cls: type[Settings],
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        class TomlSettingsSource(PydanticBaseSettingsSource):
            def __init__(self: TomlSettingsSource, s_cls: type[BaseSettings]) -> None:
                super().__init__(s_cls)

            def __call__(self: TomlSettingsSource) -> dict[str, object]:
                return _load_toml_settings()

            def get_field_value(
                self: TomlSettingsSource, field: object, field_name: str
            ) -> tuple[object, str, bool]:
                data = self()
                if field_name in data:
                    return data[field_name], field_name, True
                return None, field_name, False

        # Precedence: env vars, .env file, TOML file, init kwargs, file secrets (unused)
        return (
            env_settings,
            dotenv_settings,
            TomlSettingsSource(settings_cls),
            init_settings,
            file_secret_settings,
        )
