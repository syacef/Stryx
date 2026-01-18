"""
Ingestion Service - CPU Optimized
Main application entry point.
"""
import redis
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_QUEUE, API_HOST, API_PORT
from models import VideoRequest, VideoResponse, HealthResponse
from video_processor import process_video, get_video_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Video Ingestion Service",
    version="1.0.0",
    description="CPU-optimized video ingestion with motion detection"
)

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


@app.get("/")
async def root():
    """Root endpoint - basic service info."""
    try:
        redis_connected = redis_client.ping()
    except Exception:
        redis_connected = False
    
    return {
        "service": "Ingestion Service",
        "status": "running",
        "redis_connected": redis_connected
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check endpoint."""
    try:
        redis_connected = redis_client.ping()
        queue_length = redis_client.llen(REDIS_QUEUE)
        
        return {
            "status": "healthy",
            "redis": {
                "connected": redis_connected,
                "queue_length": queue_length
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/process", response_model=VideoResponse)
async def process_video_endpoint(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Queue a video for processing.
    
    Args:
        request: Video processing request
        background_tasks: FastAPI background tasks
    
    Returns:
        Processing status response
    """
    try:
        # Add video processing to background tasks
        background_tasks.add_task(
            process_video,
            redis_client,
            request.video_path,
            request.video_id
        )
        
        return VideoResponse(
            status="queued",
            video_path=request.video_path,
            video_id=request.video_id or "auto-generated"
        )
    except Exception as e:
        logger.error(f"Failed to queue video: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/status/{video_id}")
async def get_video_status_endpoint(video_id: str):
    """
    Get processing status for a video.
    
    Args:
        video_id: Video identifier
    
    Returns:
        Video metadata including processing statistics
    """
    metadata = get_video_status(redis_client, video_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return metadata


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
