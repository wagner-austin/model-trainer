from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


class RedisConfig(BaseSettings):
    enabled: bool = Field(default=True)
    url: str = Field(default="redis://redis:6379/0")

    model_config = {
        "extra": "ignore",
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }


class RQConfig(BaseSettings):
    queue_name: str = Field(default="training")
    job_timeout_sec: int = Field(default=86_400)  # 24h
    result_ttl_sec: int = Field(default=86_400)
    failure_ttl_sec: int = Field(default=7 * 86_400)
    retry_max: int = Field(default=1)
    retry_intervals_sec: str = Field(default="300")  # comma-separated

    model_config = {
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


class AppConfig(BaseSettings):
    data_root: str = Field(default="/data")
    artifacts_root: str = Field(default="/data/artifacts")
    runs_root: str = Field(default="/data/runs")
    logs_root: str = Field(default="/data/logs")

    model_config = {
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


class Settings(BaseSettings):
    app_env: Literal["dev", "prod"] = Field(default="dev")
    logging: LoggingConfig = LoggingConfig()
    redis: RedisConfig = RedisConfig()
    rq: RQConfig = RQConfig()
    app: AppConfig = AppConfig()

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }

