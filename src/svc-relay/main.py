import socket
import subprocess
from fastapi import FastAPI, HTTPException
import redis
from config import Config
import logging


config = Config()
node_identity = socket.gethostname()

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="RTSP Relay Service", version="1.0.0")

r = redis.Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db, decode_responses=True)
global_processes = {}

@app.post("/relay/start")
async def start_relay(stream_id: str, source_url: str):
    relay_rtsp_url = f"rtsp://0.0.0.0:{config.rtsp_port}/{stream_id}"
    
    public_url = f"rtsp://{node_identity}:{config.rtsp_port}/{stream_id}"
    
    cmd = [
        "cvlc", "-Idummy", source_url,
        "--sout", f"#rtp{{sdp={relay_rtsp_url}}}",
        ":sout-keep"
    ]
    
    try:
        global_processes[stream_id] = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        logger.info(f"Started VLC relay for stream {stream_id} from {source_url} to {relay_rtsp_url}")
        logger.info("Command used: " + " ".join(cmd))

        r.set(f"owner:{stream_id}", node_identity, ex=300) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLC Failed: {str(e)}")

    return {
        "status": "active",
        "stream_id": stream_id,
        "relay_url": public_url,
        "owner": node_identity
    }

@app.delete("/relay/stop")
async def stop_relay(stream_id: str):
    global global_processes
    try:
        r.delete(f"owner:{stream_id}")
        process = global_processes.get(stream_id)
        if process:
            process.terminate()
            process.wait()
            del global_processes[stream_id]

        logger.info(f"Stopped VLC relay for stream {stream_id}")
        return {"detail": "Relay stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        redis_connected = r.ping()
    except Exception:
        redis_connected = False
    
    return {
        "service": "RTSP Relay Service",
        "version": "1.0.0",
        "status": "running",
        "redis_connected": redis_connected
    }

@app.get("/relay/status")
async def relay_status():
    logger.info(f"Current active relays: {list(global_processes.keys())}")
    for stream_id, process in global_processes.items():
        if process.poll() is not None:
            logger.warning(f"Process for stream {stream_id} has terminated unexpectedly.")
            del global_processes[stream_id]
    return {
        "active_relays": list(global_processes.keys())
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting RTSP Relay Service on {config.api_host}:{config.api_port}")
    uvicorn.run(app, host=config.api_host, port=config.api_port)
