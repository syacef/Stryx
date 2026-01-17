"""
Video processing logic.
"""
import cv2
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import VIDEO_BUCKET, FRAME_SKIP, REDIS_QUEUE, JPEG_QUALITY
from motion_detector import MotionDetector

logger = logging.getLogger(__name__)


def process_video(redis_client, video_path: str, video_id: Optional[str] = None):
    """
    Process video file, detect motion, and send interesting frames to Redis.
    
    Args:
        redis_client: Redis client instance
        video_path: Relative path to video file (within VIDEO_BUCKET)
        video_id: Optional video identifier (auto-generated if None)
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file cannot be opened
    """
    full_path = Path(VIDEO_BUCKET) / video_path
    
    if not full_path.exists():
        logger.error(f"Video file not found: {full_path}")
        raise FileNotFoundError(f"Video file not found: {full_path}")
    
    # Generate video_id if not provided
    if video_id is None:
        video_id = f"{full_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Processing video: {full_path} (ID: {video_id})")
    
    # Open video file
    cap = cv2.VideoCapture(str(full_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {full_path}")
        raise ValueError(f"Failed to open video: {full_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize motion detector
    detector = MotionDetector()
    
    frame_count = 0
    processed_count = 0
    motion_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency (process every Nth frame)
            if frame_count % FRAME_SKIP != 0:
                continue
            
            processed_count += 1
            
            # Detect motion in frame
            has_motion, motion_score = detector.detect_motion(frame)
            
            if has_motion:
                motion_count += 1
                
                # Encode frame to JPEG for efficient storage/transmission
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_bytes = buffer.tobytes()
                
                # Prepare message for Redis queue
                message = {
                    "video_id": video_id,
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps if fps > 0 else 0,
                    "motion_score": float(motion_score),
                    "frame_data": frame_bytes.hex(),  # Hex encoding for JSON
                    "shape": frame.shape[:2],  # (height, width)
                }
                
                # Push to Redis queue for inference service
                redis_client.rpush(REDIS_QUEUE, json.dumps(message))
                
                logger.debug(f"Frame {frame_count} - Motion detected (score: {motion_score:.2f})")
    
    finally:
        cap.release()
        detector.reset()
    
    # Log processing summary
    motion_percentage = 100 * motion_count / processed_count if processed_count > 0 else 0
    logger.info(
        f"Video processing complete - "
        f"Total frames: {frame_count}, "
        f"Processed: {processed_count}, "
        f"Motion detected: {motion_count} "
        f"({motion_percentage:.1f}%)"
    )
    
    # Store processing metadata in Redis
    metadata = {
        "video_id": video_id,
        "video_path": str(video_path),
        "status": "completed",
        "total_frames": frame_count,
        "processed_frames": processed_count,
        "motion_frames": motion_count,
        "completed_at": datetime.now().isoformat()
    }
    redis_client.set(f"video_metadata:{video_id}", json.dumps(metadata))
    
    return video_id


def get_video_status(redis_client, video_id: str) -> Optional[dict]:
    """
    Get processing status for a video.
    
    Args:
        redis_client: Redis client instance
        video_id: Video identifier
    
    Returns:
        Video metadata dictionary or None if not found
    """
    metadata_key = f"video_metadata:{video_id}"
    metadata = redis_client.get(metadata_key)
    
    if not metadata:
        return None
    
    return json.loads(metadata)
