"""
Ingestion Service - CPU Optimized
Main application entry point.
"""
import redis
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_QUEUE, API_HOST, API_PORT
from models import VideoRequest, VideoResponse, HealthResponse, RTSPStreamRequest, RTSPStreamResponse, RTSPStopRequest
from video_processor import process_video, get_video_status
from rtsp_manager import RTSPStreamManager

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
    description="CPU-optimized video ingestion with motion detection and RTSP support"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Initialize RTSP Stream Manager
rtsp_manager = RTSPStreamManager(redis_client)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve frontend UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    
    # Fallback to API info
    try:
        redis_connected = redis_client.ping()
    except Exception:
        redis_connected = False
    
    return {
        "service": "Ingestion Service",
        "status": "running",
        "redis_connected": redis_connected,
        "features": ["video_processing", "rtsp_streaming"]
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


# ============================================================================
# RTSP Streaming Endpoints
# ============================================================================

@app.post("/rtsp/start", response_model=RTSPStreamResponse)
async def start_rtsp_stream(request: RTSPStreamRequest):
    """
    Start processing an RTSP stream.
    
    Args:
        request: RTSP stream request with URL and configuration
    
    Returns:
        Stream information including status
    """
    try:
        stream_info = rtsp_manager.start_stream(
            stream_url=request.stream_url,
            stream_id=request.stream_id,
            enable_motion_detection=request.enable_motion_detection
        )
        
        return RTSPStreamResponse(
            stream_id=stream_info.stream_id,
            stream_url=stream_info.stream_url,
            status=stream_info.status,
            enable_motion_detection=stream_info.enable_motion_detection,
            started_at=stream_info.started_at,
            frames_processed=stream_info.frames_processed,
            frames_sent=stream_info.frames_sent
        )
    except ValueError as e:
        logger.error(f"Failed to start stream: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error starting stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rtsp/stop")
async def stop_rtsp_stream(request: RTSPStopRequest):
    """
    Stop processing an RTSP stream.
    
    Args:
        request: Stop request with stream_id
    
    Returns:
        Final stream information
    """
    try:
        stream_info = rtsp_manager.stop_stream(request.stream_id)
        
        return {
            "stream_id": stream_info.stream_id,
            "status": stream_info.status,
            "stopped_at": stream_info.stopped_at,
            "frames_processed": stream_info.frames_processed,
            "frames_sent": stream_info.frames_sent
        }
    except ValueError as e:
        logger.error(f"Failed to stop stream: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error stopping stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rtsp/streams")
async def list_rtsp_streams():
    """
    List all RTSP streams and their status.
    
    Returns:
        Dictionary of stream_id -> stream information
    """
    try:
        streams = rtsp_manager.list_streams()
        
        # Convert StreamInfo objects to dicts
        result = {}
        for stream_id, stream_info in streams.items():
            result[stream_id] = {
                "stream_id": stream_info.stream_id,
                "stream_url": stream_info.stream_url,
                "status": stream_info.status,
                "enable_motion_detection": stream_info.enable_motion_detection,
                "started_at": stream_info.started_at,
                "stopped_at": stream_info.stopped_at,
                "frames_processed": stream_info.frames_processed,
                "frames_sent": stream_info.frames_sent,
                "error_message": stream_info.error_message
            }
        
        return {"streams": result, "total": len(result)}
    except Exception as e:
        logger.error(f"Error listing streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rtsp/streams/{stream_id}")
async def get_rtsp_stream_info(stream_id: str):
    """
    Get information about a specific RTSP stream.
    
    Args:
        stream_id: Stream identifier
    
    Returns:
        Stream information
    """
    try:
        stream_info = rtsp_manager.get_stream_info(stream_id)
        
        if stream_info is None:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
        
        return {
            "stream_id": stream_info.stream_id,
            "stream_url": stream_info.stream_url,
            "status": stream_info.status,
            "enable_motion_detection": stream_info.enable_motion_detection,
            "started_at": stream_info.started_at,
            "stopped_at": stream_info.stopped_at,
            "frames_processed": stream_info.frames_processed,
            "frames_sent": stream_info.frames_sent,
            "error_message": stream_info.error_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stream info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down, stopping all RTSP streams...")
    rtsp_manager.stop_all_streams()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
