import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    environment: str = "development"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = os.getenv("REDIS_PORT", 6379)
    redis_db: int = os.getenv("REDIS_DB", 0)

    # RTSP
    rtsp_port: int = os.getenv("RTSP_PORT", 8554)
    domain: str = os.getenv("DOMAIN", "localhost")

    # API
    api_port: int = os.getenv("API_PORT", 8000)
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
