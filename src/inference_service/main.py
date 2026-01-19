import redis
import logging
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import uvicorn
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_QUEUE,
    DEVICE, MODEL_TYPE, BATCH_TIMEOUT, API_HOST, API_PORT
)
from database import init_database, get_db_connection, get_stats
from model import load_model
from worker import process_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
redis_client = None
model = None
running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global redis_client, model, running
    
    # Startup
    logger.info("Starting inference service...")
    
    # Initialize Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    logger.info("Redis connected")
    
    # Initialize database
    init_database()
    
    # Load model
    model = load_model()
    
    running = True
    
    # Start background worker thread
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    logger.info("Background worker thread started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference service...")
    running = False
    worker_thread.join(timeout=5)


# Initialize FastAPI app
app = FastAPI(
    title="Inference Service",
    version="1.0.0",
    description="GPU-optimized inference service for video frame processing",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Inference Service",
        "status": "running",
        "device": DEVICE,
        "model_type": MODEL_TYPE
    }


@app.get("/health")
async def health():
    try:
        redis_connected = redis_client.ping()
        queue_length = redis_client.llen(REDIS_QUEUE)
        
        # Check database
        conn = get_db_connection()
        conn.close()
        db_connected = True
        
        return {
            "status": "healthy",
            "redis": {
                "connected": redis_connected,
                "queue_length": queue_length
            },
            "database": {
                "connected": db_connected
            },
            "model": {
                "loaded": model is not None,
                "device": DEVICE,
                "type": MODEL_TYPE
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/inference/process-batch")
async def process_batch_endpoint():
    try:
        count = process_batch(redis_client, model)
        return {
            "status": "completed",
            "frames_processed": count
        }
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/stats")
async def get_stats_endpoint():
    try:
        stats = get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def background_worker():
    global running, redis_client, model
    
    logger.info("Background worker started")
    
    while running:
        try:
            count = process_batch(redis_client, model)
            if count == 0:
                time.sleep(BATCH_TIMEOUT)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(BATCH_TIMEOUT)
    
    logger.info("Background worker stopped")


if __name__ == "__main__":
    # Start API server
    uvicorn.run(app, host=API_HOST, port=API_PORT)
