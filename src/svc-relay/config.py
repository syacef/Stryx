import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)

    # RTSP
    rtsp_port: int = Field(default=8554)
    domain: str = Field(default="relay.localhost")

    # API
    api_port: int = Field(default=8000)
    api_host: str = Field(default="0.0.0.0")
