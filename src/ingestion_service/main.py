import os
import redis
import logging
import uuid
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
from fastapi import UploadFile, File, Form
import shutil
import httpx

from config import Config
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

config = Config()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_manager, worker_manager, redis_client, local_worker
    
    logger.info("Starting Stream Ingestion Service...")
    redis_client = redis.Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db)

    pod_name = os.getenv("POD_NAME", f"pod-{uuid.uuid4().hex[:8]}")
    worker_id = f"worker-{pod_name}"

    stream_manager = StreamManager(redis_client, config.relay_service_url)
    worker_manager = WorkerManager(redis_client, stream_manager, config.max_streams_per_worker, config.worker_heartbeat_interval)

    local_worker = StreamWorker(worker_id, redis_client, stream_manager, config.redis_queue, config.worker_heartbeat_interval, config.jpeg_quality)
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


app = FastAPI(
    title="Stream Ingestion Service",
    version="1.0.0",
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
        stream_id = request.stream_id or f"stream_{uuid.uuid4().hex[:12]}"
        final_rtsp_url = request.rtsp_url
        public_url = request.rtsp_url

        if request.rtsp_url.startswith(('http://', 'https://')):
            logger.info(f"Detected HTTP source for {stream_id}. Requesting relay...")
            
            async with httpx.AsyncClient() as client:
                relay_response = await client.post(
                    f"{config.relay_service_url}/relay/start",
                    params={
                        "stream_id": stream_id,
                        "source_url": request.rtsp_url
                    },
                    timeout=15.0
                )
            
            if relay_response.status_code != 200:
                logger.error(f"Relay error: {relay_response.text}")
                raise HTTPException(
                    status_code=502, 
                    detail="Relay service failed to convert HTTP source to RTSP"
                )
            
            relay_data = relay_response.json()
            final_rtsp_url = relay_data.get("relay_url")
            public_url = relay_data.get("public_url")
        
        if not final_rtsp_url.startswith(('rtsp://', 'rtsps://')):
            raise HTTPException(
                status_code=400, 
                detail="Source must be RTSP or a valid HTTP MP4 link"
            )

        country, continent = request.country, request.continent
        if request.latitude is not None and request.longitude is not None:
            if not country or not continent:
                try:
                    calc_country, calc_continent = get_location_details(request.latitude, request.longitude)
                    country, continent = calc_country, calc_continent
                except Exception as e:
                    logger.warning(f"Geolocation failed: {e}")

        result = stream_manager.register_stream(
            stream_id=stream_id,
            rtsp_url=final_rtsp_url,
            public_url=public_url,
            name=request.name,
            source="direct" if request.rtsp_url.startswith(('rtsp://', 'rtsps://')) else "relay",
            country=country,
            continent=continent
        )

        if not result:
            raise HTTPException(status_code=400, detail="Stream already exists")
        
        worker_id = worker_manager.assign_worker_to_stream(stream_id)
        
        return StreamResponse(
            stream_id=stream_id,
            rtsp_url=final_rtsp_url,
            name=request.name,
            status="registered",
            worker_id=worker_id,
            country=country,
            continent=continent
        )

    except Exception as e:
        logger.error(f"Failed to register: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/streams/upload")
async def upload_video(
    video: UploadFile = File(...),
    name: str = Form(...)
):
    try:
        file_id = uuid.uuid4().hex[:8]
        file_path = UPLOAD_DIR / f"{file_id}_{video.filename}"
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        internal_rtsp_url = f"rtsp://localhost:8554/{file_id}"

        stream_id = f"stream_{file_id}"
        result = stream_manager.register_stream(
            stream_id=stream_id,
            rtsp_url=internal_rtsp_url,
            name=name,
            source="relay" # Tagging as relay for the frontend
        )

        if not result:
            raise HTTPException(status_code=400, detail="Stream registration failed")

        worker_id = worker_manager.assign_worker_to_stream(stream_id)

        return {
            "streamId": stream_id,
            "rtspUrl": internal_rtsp_url,
            "status": "success",
            "worker_id": worker_id
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/streams/{stream_id}", response_model=StreamDeleteResponse)
async def delete_stream(stream_id: str):
    try:
        stream_info = stream_manager.get_stream_info(stream_id)
        
        if not stream_info:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
        
        worker_manager.unassign_worker_from_stream(stream_id)
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
    import uvicorn
    uvicorn.run(app, host=config.api_host, port=config.api_port)
