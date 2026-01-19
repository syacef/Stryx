import json
import logging
from typing import Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class StreamManager:

    STREAM_PREFIX = "stream:"
    STREAM_LIST_KEY = "streams:active"

    def __init__(self, redis_client):
        self.redis = redis_client
        logger.info("StreamManager initialized")
    
    def register_stream(
        self, 
        stream_id: str, 
        rtsp_url: str,
        name: Optional[str] = None,
        fps: int = 5,
        resolution: Optional[tuple[int, int]] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        country: Optional[str] = None,
        continent: Optional[str] = None,
        source : str = "direct"
    ) -> bool:
        stream_key = f"{self.STREAM_PREFIX}{stream_id}"
        
        if self.redis.exists(stream_key):
            logger.warning(f"Stream {stream_id} already exists")
            return False
        
        stream_data = {
            "stream_id": stream_id,
            "rtsp_url": rtsp_url,
            "name": name or stream_id,
            "fps": fps,
            "resolution": resolution,
            "latitude": latitude,
            "longitude": longitude,
            "country": country,
            "continent": continent,
            "status": "registered",
            "source": source,
            "created_at": datetime.now().isoformat(),
            "frame_count": 0,
            "last_frame_at": None,
            "worker_id": None
        }

        self.redis.set(stream_key, json.dumps(stream_data))
        
        self.redis.sadd(self.STREAM_LIST_KEY, stream_id)
        
        logger.info(f"Stream {stream_id} registered successfully")
        return True
    
    def delete_stream(self, stream_id: str) -> bool:
        stream_key = f"{self.STREAM_PREFIX}{stream_id}"
        
        if not self.redis.exists(stream_key):
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        source = self.get_stream_info(stream_id).get("source", False)
        if source == "relay":
            try:
                response = requests.delete(f"{relay_url}/relay/stop", params={"stream_id": stream_id})
                if response.status_code != 200:
                    logger.error(f"Failed to stop relay for stream {stream_id}: {response.text}")
            except Exception as e:
                logger.error(f"Error stopping relay for stream {stream_id}: {e}")
                return False

        self.redis.delete(stream_key)
        
        self.redis.srem(self.STREAM_LIST_KEY, stream_id)
        
        logger.info(f"Stream {stream_id} deleted successfully")
        return True
    
    def get_stream_info(self, stream_id: str) -> Optional[dict]:
        stream_key = f"{self.STREAM_PREFIX}{stream_id}"
        stream_data = self.redis.get(stream_key)
        
        if not stream_data:
            return None
        
        return json.loads(stream_data)
    
    def update_stream_status(self, stream_id: str, status: str):
        stream_info = self.get_stream_info(stream_id)
        if stream_info:
            stream_info["status"] = status
            stream_info["updated_at"] = datetime.now().isoformat()
            stream_key = f"{self.STREAM_PREFIX}{stream_id}"
            self.redis.set(stream_key, json.dumps(stream_info))
    
    def update_worker_assignment(self, stream_id: str, worker_id: Optional[str]):
        stream_info = self.get_stream_info(stream_id)
        if stream_info:
            stream_info["worker_id"] = worker_id
            stream_info["updated_at"] = datetime.now().isoformat()
            stream_key = f"{self.STREAM_PREFIX}{stream_id}"
            self.redis.set(stream_key, json.dumps(stream_info))
    
    def increment_frame_count(self, stream_id: str):
        stream_info = self.get_stream_info(stream_id)
        if stream_info:
            stream_info["frame_count"] += 1
            stream_info["last_frame_at"] = datetime.now().isoformat()
            stream_key = f"{self.STREAM_PREFIX}{stream_id}"
            self.redis.set(stream_key, json.dumps(stream_info))
    
    def list_all_streams(self) -> list[dict]:
        stream_ids = self.redis.smembers(self.STREAM_LIST_KEY)
        streams = []
        
        for stream_id in stream_ids:
            stream_id_str = stream_id.decode('utf-8') if isinstance(stream_id, bytes) else stream_id
            stream_info = self.get_stream_info(stream_id_str)
            if stream_info:
                streams.append(stream_info)
        
        return streams
    
    def get_active_stream_count(self) -> int:
        return self.redis.scard(self.STREAM_LIST_KEY)
    
    def cleanup(self):
        logger.info("StreamManager cleanup complete")
