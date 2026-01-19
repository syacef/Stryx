import cv2
import json
import logging
import time
import uuid
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class StreamWorker:
    
    def __init__(self, worker_id: str, redis_client, stream_manager, redis_queue: str, worker_heartbeat_interval: int = 30, jpeg_quality: int = 80):
        self.worker_id = worker_id
        self.redis = redis_client
        self.stream_manager = stream_manager
        self.active_streams = {}  # stream_id -> VideoCapture
        self.running = False
        self.redis_queue = redis_queue
        self.worker_heartbeat_interval = worker_heartbeat_interval
        self.jpeg_quality = jpeg_quality
        self.worker_thread = None
        logger.info(f"Worker {worker_id} initialized")
    
    def start_async(self):
        if self.running:
            logger.warning(f"Worker {self.worker_id} already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
        logger.info(f"Worker {self.worker_id} started in background")
    
    def _run(self):
        logger.info(f"Worker {self.worker_id} starting processing loop")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                current_time = time.time()

                if current_time - last_heartbeat >= self.worker_heartbeat_interval:
                    self._send_heartbeat()
                    last_heartbeat = current_time
                
                assigned_streams = self._get_assigned_streams()
                
                for stream_id in assigned_streams:
                    if stream_id not in self.active_streams:
                        self._start_stream(stream_id)
                
                for stream_id in list(self.active_streams.keys()):
                    if stream_id not in assigned_streams:
                        self._stop_stream(stream_id)
                
                self._process_frames()
                
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)
    
    def stop(self):
        logger.info(f"Stopping worker {self.worker_id}")
        self.running = False
        
        # Stop all active streams
        for stream_id in list(self.active_streams.keys()):
            self._stop_stream(stream_id)
        
        # Wait for thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _send_heartbeat(self):
        try:
            worker_key = f"worker:{self.worker_id}"
            worker_data = self.redis.get(worker_key)
            
            if worker_data:
                worker_info = json.loads(worker_data)
                worker_info["last_heartbeat"] = datetime.now().isoformat()
                worker_info["last_heartbeat_timestamp"] = time.time()
                worker_info["active_stream_count"] = len(self.active_streams)
                self.redis.set(worker_key, json.dumps(worker_info))
                self.redis.expire(worker_key, self.worker_heartbeat_interval * 3)  # TTL
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
    
    def _get_assigned_streams(self) -> set[str]:
        try:
            worker_streams_key = f"worker_streams:{self.worker_id}"
            stream_ids = self.redis.smembers(worker_streams_key)
            return {
                s.decode('utf-8') if isinstance(s, bytes) else s for s in stream_ids
            }
        except Exception as e:
            logger.error(f"Error getting assigned streams: {e}")
            return set()
    
    def _start_stream(self, stream_id: str):
        try:
            # Get stream info
            stream_info = self.stream_manager.get_stream_info(stream_id)
            
            if not stream_info:
                logger.error(f"Stream {stream_id} not found in Redis")
                return
            
            rtsp_url = stream_info["rtsp_url"]
            
            # Open RTSP stream
            logger.info(f"Opening stream {stream_id}: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            
            # Set buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.error(f"Failed to open stream {stream_id}")
                self.stream_manager.update_stream_status(stream_id, "error")
                return
            
            # Store stream info
            self.active_streams[stream_id] = {
                "capture": cap,
                "info": stream_info,
                "frame_count": 0,
                "last_frame_time": time.time(),
                "target_fps": stream_info.get("fps", 5),
                "resolution": stream_info.get("resolution"),
                "reconnect_attempts": 0,
                "last_reconnect": 0,
                "prev_gray": None
            }
            
            self.stream_manager.update_stream_status(stream_id, "active")
            
            logger.info(f"Stream {stream_id} started successfully")
            
        except Exception as e:
            logger.error(f"Error starting stream {stream_id}: {e}")
            self.stream_manager.update_stream_status(stream_id, "error")
    
    def _stop_stream(self, stream_id: str):
        if stream_id not in self.active_streams:
            return
        
        try:
            stream_data = self.active_streams[stream_id]
            stream_data["capture"].release()
            del self.active_streams[stream_id]
            
            self.stream_manager.update_stream_status(stream_id, "stopped")
            
            logger.info(f"Stream {stream_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping stream {stream_id}: {e}")
    
    def _process_frames(self):
        current_time = time.time()
        
        for stream_id, stream_data in list(self.active_streams.items()):
            try:
                target_fps = stream_data["target_fps"]
                frame_interval = 1.0 / target_fps
                
                if current_time - stream_data["last_frame_time"] < frame_interval:
                    continue
                
                # Capture frame
                cap = stream_data["capture"]
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from stream {stream_id}")
                    
                    # Attempt reconnection with backoff
                    if current_time - stream_data["last_reconnect"] > 5:  # Wait 5s between reconnects
                        stream_data["reconnect_attempts"] += 1
                        stream_data["last_reconnect"] = current_time
                        
                        if stream_data["reconnect_attempts"] <= 3:
                            logger.info(f"Attempting to reconnect stream {stream_id} (attempt {stream_data['reconnect_attempts']})")
                            self._stop_stream(stream_id)
                            self._start_stream(stream_id)
                        else:
                            logger.error(f"Stream {stream_id} exceeded reconnection attempts, marking as error")
                            self.stream_manager.update_stream_status(stream_id, "error")
                    
                    continue
                
                stream_data["reconnect_attempts"] = 0
                
                resolution = stream_data["resolution"]
                if resolution:
                    frame = cv2.resize(frame, resolution)
                
                # Process and send frame
                self._send_frame(stream_id, frame, stream_data)
                
                stream_data["last_frame_time"] = current_time
                stream_data["frame_count"] += 1
                
            except Exception as e:
                logger.error(f"Error processing frame from stream {stream_id}: {e}")
    
    def _send_frame(self, stream_id: str, frame, stream_data: dict):
        try:
            # Calculate motion score
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            
            if stream_data["prev_gray"] is not None:
                frame_diff = cv2.absdiff(stream_data["prev_gray"], gray)
                motion_score = float(frame_diff.mean())
            
            stream_data["prev_gray"] = gray
            
            frame_id = f"{stream_id}_{uuid.uuid4().hex[:8]}"

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            frame_bytes = buffer.tobytes()
            
            # Calculate timestamp
            frame_count = stream_data["frame_count"]
            target_fps = stream_data["target_fps"]
            timestamp = frame_count / target_fps if target_fps > 0 else 0
            
            # Create frame message
            stream_info = stream_data["info"]
            message = {
                "stream_id": stream_id,
                "video_id": stream_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "frame_number": frame_count,
                "frame_data": frame_bytes.hex(),
                "motion_score": motion_score,
                "shape": frame.shape[:2],  # (height, width)
                "worker_id": self.worker_id,
                "captured_at": datetime.now().isoformat(),
                "latitude": stream_info.get("latitude"),
                "longitude": stream_info.get("longitude"),
                "country": stream_info.get("country"),
                "continent": stream_info.get("continent")
            }
            
            # Push to Redis queue
            self.redis.rpush(self.redis_queue, json.dumps(message))

            self.stream_manager.increment_frame_count(stream_id)
            
            logger.debug(f"Frame {frame_id} from stream {stream_id} sent to queue")
            
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
