import os
from typing import Optional
from pydantic_settings import BaseSettings


class ServiceConfig(BaseSettings):
    """Base configuration for all services."""
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_queue: str = "frame_queue"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class IngestionConfig(ServiceConfig):
    """Configuration for ingestion service."""
    
    # Video processing
    video_bucket: str = "/data/videos"
    frame_skip: int = 5
    
    # Motion detection
    motion_threshold: float = 25.0
    min_motion_area: int = 500
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


class InferenceConfig(ServiceConfig):
    """Configuration for inference service."""
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "safari"
    db_user: str = "postgres"
    db_password: str = "postgres"
    
    # Model
    model_type: str = "classifier"  # "classifier" or "ssl"
    model_checkpoint: Optional[str] = None
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Batching
    batch_size: int = 8
    batch_timeout: float = 2.0
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    
    @property
    def db_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
