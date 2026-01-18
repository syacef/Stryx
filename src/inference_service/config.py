import os
import torch

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "frame_queue")

# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "safari")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Model configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "dummy")  # "dummy", "classifier", or "ssl"
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", None)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch processing configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 2.0))  # seconds

# Service configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8001))
