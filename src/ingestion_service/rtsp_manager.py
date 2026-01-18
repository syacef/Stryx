"""
RTSP Stream Manager
Handles real-time video streams from cameras.
"""
import cv2
import json
import logging
import threading
import time
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from config import FRAME_SKIP, REDIS_QUEUE, JPEG_QUALITY
from motion_detector import MotionDetector

logger = logging.getLogger(__name__)


@dataclass
class StreamInfo:
    """Information about an RTSP stream."""
    stream_id: str
    stream_url: str
    status: str  # 'starting', 'running', 'stopped', 'error'
    enable_motion_detection: bool
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    frames_processed: int = 0
    frames_sent: int = 0
    error_message: Optional[str] = None


class RTSPStreamManager:
    """Manages multiple RTSP streams."""
    
    def __init__(self, redis_client):
        """
        Initialize RTSP stream manager.
        
        Args:
            redis_client: Redis client for frame queue
        """
        self.redis_client = redis_client
        self.streams: Dict[str, StreamInfo] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        logger.info("RTSP Stream Manager initialized")
    
    def start_stream(
        self, 
        stream_url: str, 
        stream_id: str, 
        enable_motion_detection: bool = True
    ) -> StreamInfo:
        """
        Start processing an RTSP stream.
        
        Args:
            stream_url: RTSP stream URL (e.g., rtsp://camera-ip:554/stream1)
            stream_id: Unique identifier for the stream
            enable_motion_detection: Whether to use motion detection filtering
        
        Returns:
            StreamInfo object with stream details
        
        Raises:
            ValueError: If stream_id already exists and is running
        """
        with self._lock:
            # Check if stream already exists
            if stream_id in self.streams:
                existing = self.streams[stream_id]
                if existing.status in ['starting', 'running']:
                    raise ValueError(f"Stream {stream_id} is already running")
            
            # Create stream info
            stream_info = StreamInfo(
                stream_id=stream_id,
                stream_url=stream_url,
                status='starting',
                enable_motion_detection=enable_motion_detection,
                started_at=datetime.now().isoformat()
            )
            
            self.streams[stream_id] = stream_info
            
            # Create stop event
            stop_event = threading.Event()
            self.stop_events[stream_id] = stop_event
            
            # Start processing thread
            thread = threading.Thread(
                target=self._process_stream,
                args=(stream_id, stream_url, enable_motion_detection, stop_event),
                daemon=True,
                name=f"RTSP-{stream_id}"
            )
            self.stream_threads[stream_id] = thread
            thread.start()
            
            logger.info(f"Started RTSP stream: {stream_id} ({stream_url})")
            return stream_info
    
    def stop_stream(self, stream_id: str) -> StreamInfo:
        """
        Stop processing an RTSP stream.
        
        Args:
            stream_id: Stream identifier
        
        Returns:
            StreamInfo object with final stream details
        
        Raises:
            ValueError: If stream_id doesn't exist
        """
        with self._lock:
            if stream_id not in self.streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Signal thread to stop
            if stream_id in self.stop_events:
                self.stop_events[stream_id].set()
            
            # Update status
            stream_info = self.streams[stream_id]
            stream_info.status = 'stopped'
            stream_info.stopped_at = datetime.now().isoformat()
            
            logger.info(f"Stopped RTSP stream: {stream_id}")
            return stream_info
    
    def get_stream_info(self, stream_id: str) -> Optional[StreamInfo]:
        """
        Get information about a stream.
        
        Args:
            stream_id: Stream identifier
        
        Returns:
            StreamInfo object or None if not found
        """
        return self.streams.get(stream_id)
    
    def list_streams(self) -> Dict[str, StreamInfo]:
        """
        List all streams.
        
        Returns:
            Dictionary of stream_id -> StreamInfo
        """
        return self.streams.copy()
    
    def _process_stream(
        self, 
        stream_id: str, 
        stream_url: str, 
        enable_motion_detection: bool,
        stop_event: threading.Event
    ):
        """
        Process frames from an RTSP stream in a background thread.
        
        Args:
            stream_id: Stream identifier
            stream_url: RTSP URL
            enable_motion_detection: Whether to filter with motion detection
            stop_event: Event to signal thread should stop
        """
        motion_detector = MotionDetector() if enable_motion_detection else None
        cap = None
        frame_count = 0
        sent_count = 0
        
        try:
            # Open RTSP stream
            logger.info(f"Opening RTSP stream: {stream_url}")
            cap = cv2.VideoCapture(stream_url)
            
            # Configure stream settings for better stability
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open RTSP stream: {stream_url}")
            
            # Update status to running
            with self._lock:
                if stream_id in self.streams:
                    self.streams[stream_id].status = 'running'
            
            logger.info(f"RTSP stream opened successfully: {stream_id}")
            
            # Process frames
            while not stop_event.is_set():
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from {stream_id}, attempting reconnection...")
                    # Try to reconnect
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(stream_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not cap.isOpened():
                        raise ValueError(f"Failed to reconnect to {stream_url}")
                    continue
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % FRAME_SKIP != 0:
                    continue
                
                # Motion detection
                should_send = True
                if motion_detector is not None:
                    motion_detected = motion_detector.detect(frame)
                    should_send = motion_detected
                
                if should_send:
                    # Encode frame as JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    # Prepare metadata
                    metadata = {
                        'stream_id': stream_id,
                        'frame_number': frame_count,
                        'timestamp': datetime.now().isoformat(),
                        'source_type': 'rtsp',
                        'stream_url': stream_url
                    }
                    
                    # Send to Redis
                    frame_data = {
                        'image': buffer.tobytes(),
                        'metadata': json.dumps(metadata)
                    }
                    
                    self.redis_client.rpush(
                        REDIS_QUEUE,
                        json.dumps({
                            'metadata': metadata,
                            'image_size': len(buffer.tobytes())
                        })
                    )
                    
                    sent_count += 1
                    
                    # Update stats
                    with self._lock:
                        if stream_id in self.streams:
                            self.streams[stream_id].frames_processed = frame_count
                            self.streams[stream_id].frames_sent = sent_count
                    
                    if sent_count % 100 == 0:
                        logger.info(
                            f"Stream {stream_id}: processed={frame_count}, sent={sent_count}"
                        )
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error processing stream {stream_id}: {e}")
            with self._lock:
                if stream_id in self.streams:
                    self.streams[stream_id].status = 'error'
                    self.streams[stream_id].error_message = str(e)
                    self.streams[stream_id].stopped_at = datetime.now().isoformat()
        
        finally:
            # Cleanup
            if cap is not None:
                cap.release()
            
            logger.info(f"Stream {stream_id} stopped. Processed: {frame_count}, Sent: {sent_count}")
            
            # Update final stats
            with self._lock:
                if stream_id in self.streams:
                    self.streams[stream_id].frames_processed = frame_count
                    self.streams[stream_id].frames_sent = sent_count
                    if self.streams[stream_id].status == 'running':
                        self.streams[stream_id].status = 'stopped'
                        self.streams[stream_id].stopped_at = datetime.now().isoformat()
    
    def stop_all_streams(self):
        """Stop all running streams."""
        with self._lock:
            stream_ids = list(self.streams.keys())
        
        for stream_id in stream_ids:
            try:
                self.stop_stream(stream_id)
            except Exception as e:
                logger.error(f"Error stopping stream {stream_id}: {e}")
        
        logger.info("All streams stopped")
