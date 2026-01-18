import os
import redis
import logging
import uuid
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, API_HOST, API_PORT
from models import (
    StreamRegisterRequest, StreamResponse, StreamDeleteResponse,
    HealthResponse, WorkerStatusResponse
)
from stream_manager import StreamManager
from worker_manager import WorkerManager
from stream_worker import StreamWorker
from geolocation import get_location_details

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
stream_manager = None
worker_manager = None
redis_client = None
local_worker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global stream_manager, worker_manager, redis_client, local_worker
    
    # Startup
    logger.info("Starting Stream Ingestion Service...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    
    # Generate unique worker ID for this pod
    pod_name = os.getenv("POD_NAME", f"pod-{uuid.uuid4().hex[:8]}")
    worker_id = f"worker-{pod_name}"
    
    stream_manager = StreamManager(redis_client)
    worker_manager = WorkerManager(redis_client, stream_manager)
    
    local_worker = StreamWorker(worker_id, redis_client, stream_manager)
    local_worker.start_async()

    worker_manager.register_worker(worker_id)
    worker_manager.start()
    
    logger.info(f"Service started with worker ID: {worker_id}")
    
    yield
    
    logger.info("Shutting down Stream Ingestion Service...")
    
    if local_worker:
        local_worker.stop()
    worker_manager.unregister_worker(worker_id)
    worker_manager.stop()
    stream_manager.cleanup()


# Initialize FastAPI app
app = FastAPI(
    title="Stream Ingestion Service",
    version="2.0.0",
    description="RTSP stream management with distributed worker allocation",
    lifespan=lifespan
)


@app.get("/")
async def root():
    try:
        redis_connected = redis_client.ping()
    except Exception:
        redis_connected = False
    
    return {
        "service": "Stream Ingestion Service",
        "version": "1.0.0",
        "status": "running",
        "redis_connected": redis_connected
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        redis_connected = redis_client.ping()
        active_streams = stream_manager.get_active_stream_count()
        active_workers = worker_manager.get_active_worker_count()
        
        return {
            "status": "healthy",
            "redis": {
                "connected": redis_connected,
                "active_streams": active_streams,
                "active_workers": active_workers
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/streams/register", response_model=StreamResponse)
async def register_stream(request: StreamRegisterRequest):
    try:
        # Generate unique stream ID
        stream_id = request.stream_id or f"stream_{uuid.uuid4().hex[:12]}"
        
        # Validate RTSP URL
        if not request.rtsp_url.startswith(('rtsp://', 'rtsps://')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid RTSP URL. Must start with rtsp:// or rtsps://"
            )
            
        # Enrich location data if needed
        country = request.country
        continent = request.continent
        
        if request.latitude is not None and request.longitude is not None:
            if not country or not continent:
                try:
                    calc_country, calc_continent = get_location_details(request.latitude, request.longitude)
                    country = country or calc_country
                    continent = continent or calc_continent
                except Exception as e:
                    logger.warning(f"Failed to calculate location details: {e}")
        
        # Register stream
        result = stream_manager.register_stream(
            stream_id=stream_id,
            rtsp_url=request.rtsp_url,
            name=request.name,
            fps=request.fps,
            resolution=request.resolution,
            latitude=request.latitude,
            longitude=request.longitude,
            country=country,
            continent=continent
        )
        
        if not result:
            raise HTTPException(
                status_code=400, 
                detail=f"Stream {stream_id} already exists"
            )
        
        worker_id = worker_manager.assign_worker_to_stream(stream_id)
        
        logger.info(f"Stream {stream_id} registered and assigned to worker {worker_id}")
        
        return StreamResponse(
            stream_id=stream_id,
            rtsp_url=request.rtsp_url,
            name=request.name,
            status="registered",
            worker_id=worker_id,
            latitude=request.latitude,
            longitude=request.longitude,
            country=country,
            continent=continent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/streams/{stream_id}", response_model=StreamDeleteResponse)
async def delete_stream(stream_id: str):
    try:
        # Get stream info before deletion
        stream_info = stream_manager.get_stream_info(stream_id)
        
        if not stream_info:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
        
        # Unassign worker
        worker_manager.unassign_worker_from_stream(stream_id)
        
        # Delete stream
        result = stream_manager.delete_stream(stream_id)
        
        if not result:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to delete stream {stream_id}"
            )
        
        logger.info(f"Stream {stream_id} deleted successfully")
        
        return StreamDeleteResponse(
            stream_id=stream_id,
            status="deleted",
            message=f"Stream {stream_id} has been deleted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streams/{stream_id}")
async def get_stream_info(stream_id: str):
    stream_info = stream_manager.get_stream_info(stream_id)
    
    if not stream_info:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    return stream_info


@app.get("/streams")
async def list_streams():
    return stream_manager.list_all_streams()


@app.get("/workers", response_model=list[WorkerStatusResponse])
async def list_workers():
    return worker_manager.get_worker_status()


@app.get("/workers/{worker_id}")
async def get_worker_info(worker_id: str):
    worker_info = worker_manager.get_worker_info(worker_id)
    if not worker_info:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
    return worker_info


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
