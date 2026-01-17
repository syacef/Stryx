"""
Configuration for ingestion service.
"""
import os

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "frame_queue")

# Motion detection configuration
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", 25.0))
MIN_MOTION_AREA = int(os.getenv("MIN_MOTION_AREA", 500))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 5))  # Process every Nth frame

# Video configuration
VIDEO_BUCKET = os.getenv("VIDEO_BUCKET", "/data/videos")
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 85))

# Service configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
