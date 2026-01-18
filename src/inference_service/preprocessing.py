import numpy as np
import torch
import cv2
from torchvision import transforms
from typing import List


def decode_frame(frame_hex: str, shape: List[int]) -> np.ndarray:
    """
    Decode frame from hex string.
    
    Args:
        frame_hex: Hex-encoded JPEG frame
        shape: Original frame shape [height, width]
    
    Returns:
        Decoded frame as numpy array
    """
    frame_bytes = bytes.fromhex(frame_hex)
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    
    # Decode JPEG
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    
    return frame


def preprocess_frames(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocess frames for model input.
    
    Args:
        frames: List of numpy arrays (H, W, C) in BGR format
    
    Returns:
        Tensor of shape (B, C, H, W) in RGB format, normalized
    """
    # Define preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each frame
    tensors = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(frame_rgb)
        tensors.append(tensor)
    
    # Stack into batch
    batch = torch.stack(tensors)
    return batch
