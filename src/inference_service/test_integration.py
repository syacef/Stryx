import redis
import json
import numpy as np
import cv2
import requests
import time
import argparse
import sys
import uuid
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_dummy_frame():
    """Generate a random colored frame and encode as JPEG hex string."""
    # Create random image (H, W, C)
    height, width = 224, 224
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    # Convert to hex string
    hex_data = buffer.tobytes().hex()
    
    return hex_data, [height, width]

def run_test(args):
    """Run the integration test."""
    
    # 1. Check Health
    base_url = f"http://{args.api_host}:{args.api_port}"
    logger.info(f"Checking service health at {base_url}/health...")
    
    try:
        resp = requests.get(f"{base_url}/health")
        resp.raise_for_status()
        health_data = resp.json()
        logger.info(f"Health Status: {json.dumps(health_data, indent=2)}")
        
        if not health_data.get("status") == "healthy":
            logger.error("Service is not healthy!")
            return False
            
    except Exception as e:
        logger.error(f"Failed to contact service: {e}")
        return False

    # 2. Connect to Redis
    logger.info(f"Connecting to Redis at {args.redis_host}:{args.redis_port}...")
    try:
        r = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db)
        r.ping()
        logger.info("Redis connected.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return False

    # 3. Push dummy data to Redis
    video_id = f"test_video_{uuid.uuid4().hex[:8]}"
    frame_count = 5
    logger.info(f"Pushing {frame_count} dummy frames for video_id: {video_id}")
    
    for i in range(frame_count):
        hex_data, shape = generate_dummy_frame()
        
        message = {
            "video_id": video_id,
            "frame_number": i,
            "timestamp": float(i) * 0.1,
            "motion_score": 0.5,
            "frame_data": hex_data,
            "shape": shape
        }
        
        r.rpush(args.redis_queue, json.dumps(message))
    
    logger.info(f"Pushed {frame_count} frames to queue '{args.redis_queue}'")
    
    # 4. Trigger processing
    logger.info("Triggering batch processing...")
    try:
        start_time = time.time()
        # We might need to call it multiple times if batch size > frame_count or dependent on worker
        # But here we just call it once to see if it picks up something
        resp = requests.post(f"{base_url}/process-batch")
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"Process Batch Result: {result}")
        
        processed = result.get("frames_processed", 0)
        logger.info(f"Processed {processed} frames in {time.time() - start_time:.2f}s")
        
        # If the background worker is running, it might have picked them up before we called process-batch
        # So we should check stats as well
        
    except Exception as e:
        logger.error(f"Failed to trigger batch processing: {e}")
        return False

    # 5. Check stats
    logger.info("Checking stats...")
    try:
        resp = requests.get(f"{base_url}/stats")
        stats = resp.json()
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        
        # Basic verification logic
        # Note: If running against a fresh DB, we expect to see our frames.
        # But if shared DB, numbers might be higher.
        if stats.get("total_frames", 0) > 0:
            logger.info("SUCCESS: Stats show frames have been processed.")
            return True
        else:
            logger.warning("WARNING: Stats show 0 processed frames. (Maybe worker picked them up but DB async write?)")
            return False
            
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Inference Service Integration")
    
    parser.add_argument("--api-host", default="localhost", help="API Host")
    parser.add_argument("--api-port", type=int, default=8001, help="API Port")
    parser.add_argument("--redis-host", default="localhost", help="Redis Host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis Port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB")
    parser.add_argument("--redis-queue", default="frame_queue", help="Redis Queue Name")
    
    args = parser.parse_args()
    
    success = run_test(args)
    sys.exit(0 if success else 1)
