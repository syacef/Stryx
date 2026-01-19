import socket
import subprocess
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import redis
import yt_dlp
from config import Config
import threading

config = Config()
node_identity = socket.gethostname()

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="RTSP Relay Service", version="1.0.0")

r = redis.Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db, decode_responses=True)
global_processes = {}

async def cleanup_orphans():
    while True:
        await asyncio.sleep(30)
        logger.debug("Running orphan cleanup check...")

        for stream_id in list(global_processes.keys()):
            owner = r.get(f"owner:{stream_id}")
            
            if owner != node_identity:
                logger.warning(f"Stream {stream_id} no longer owned by this node. Terminating.")
                stop_vlc_process(stream_id)

def stop_vlc_process(stream_id: str):
    process = global_processes.get(stream_id)
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating VLC for {stream_id}: {e}")
        finally:
            global_processes.pop(stream_id, None)

def log_vlc_output(pipe, stream_id):
    with pipe:
        for line in iter(pipe.readline, b''):
            msg = line.decode().strip()
            if msg:
                logger.info(f"VLC [{stream_id}]: {msg}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_orphans())
    yield
    logger.info("Shutdown signal received. Cleaning up all relays...")
    cleanup_task.cancel()

    for stream_id in list(global_processes.keys()):
        r.delete(f"owner:{stream_id}")
        stop_vlc_process(stream_id)
    logger.info("Cleanup complete.")

def get_actual_url(source_url: str):
    youtube_domains = ['youtube.com', 'youtu.be']
    if any(domain in source_url for domain in youtube_domains):
        logger.info(f"Extracting direct stream URL for YouTube link: {source_url}")
        ydl_opts = {
            'format': 'best',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_url, download=False)
                return info['url']
        except Exception as e:
            logger.error(f"yt-dlp extraction failed: {str(e)}")
            raise Exception(f"Could not extract YouTube stream: {str(e)}")
    return source_url

@app.post("/relay/start")
async def start_relay(stream_id: str, source_url: str):
    try:
        actual_source = get_actual_url(source_url)

        relay_rtsp_url = f"rtsp://0.0.0.0:{config.rtsp_port}/{stream_id}"
        public_url = f"rtsp://{config.domain}:{config.rtsp_port}/{stream_id}"
        internal_url = f"rtsp://{node_identity}:{config.rtsp_port}/{stream_id}"

        sout_config = f"#rtp{{sdp={relay_rtsp_url}}}"

        cmd = [
            "cvlc", "-Idummy", actual_source,
            "--network-caching=1500",
            "--sout", sout_config,
            "--sout-keep"
        ]

        global_processes[stream_id] = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
        )

        if global_processes[stream_id].poll() is not None:
            logger.error(f"VLC process for stream {stream_id} terminated unexpectedly.")
            raise Exception("Failed to start VLC process.")

        threading.Thread(target=log_vlc_output, args=(global_processes[stream_id].stdout, stream_id), daemon=True).start()
        threading.Thread(target=log_vlc_output, args=(global_processes[stream_id].stderr, stream_id), daemon=True).start()

        logger.info(f"Started VLC relay for stream {stream_id}. Source Type: {'YouTube' if actual_source != source_url else 'Direct'}")
        logger.info("Relay Command: " + " ".join(cmd))
        r.set(f"owner:{stream_id}", node_identity, ex=300) 
    except Exception as e:
        logger.error(f"Relay start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "active",
        "stream_id": stream_id,
        "relay_url": internal_url,
        "public_url": public_url,
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
        return
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
    global global_processes
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
