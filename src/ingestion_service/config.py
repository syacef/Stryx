import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    redis_db: int = int(os.getenv("REDIS_DB", 0))
    redis_queue: str = os.getenv("REDIS_QUEUE", "frame_queue")

    max_streams_per_worker: int = int(os.getenv("MAX_STREAMS_PER_WORKER", 5))
    worker_heartbeat_interval: int = int(os.getenv("WORKER_HEARTBEAT_INTERVAL", 30))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", 80))

    relay_service_url: str = os.getenv("RELAY_SERVICE_URL", "http://svc-relay:8000")

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
