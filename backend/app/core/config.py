from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="LotoSystem")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_prefix: str = Field(default="/api")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOTO_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
