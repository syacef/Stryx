import json
import logging
from typing import List, Dict
from config import REDIS_QUEUE, BATCH_SIZE
from preprocessing import decode_frame
from model import run_inference
from database import save_results

logger = logging.getLogger(__name__)


def process_batch(redis_client, model) -> int:
    """
    Process a batch of frames from Redis queue.
    
    Args:
        redis_client: Redis client instance
        model: PyTorch model for inference
    
    Returns:
        Number of frames processed
    """
    # Fetch batch from queue
    messages = []
    frames = []
    
    for _ in range(BATCH_SIZE):
        data = redis_client.lpop(REDIS_QUEUE)
        if data is None:
            break
        
        try:
            message = json.loads(data)
            frame = decode_frame(message["frame_data"], message["shape"])
            
            messages.append(message)
            frames.append(frame)
        
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            continue
    
    if not frames:
        return 0
    
    # Run inference
    try:
        predictions = run_inference(model, frames)
        
        # Save results
        save_results(messages, predictions)
        
        logger.info(f"Processed batch of {len(frames)} frames")
        return len(frames)
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        # Re-queue messages
        for msg in messages:
            redis_client.rpush(REDIS_QUEUE, json.dumps(msg))
        return 0
