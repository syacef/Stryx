
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any
from config import MODEL_TYPE, MODEL_CHECKPOINT, DEVICE
from preprocessing import preprocess_frames

logger = logging.getLogger(__name__)


class DummyModel(nn.Module):
    """
    Dummy model for testing without real model dependencies.
    Returns random predictions based on model type.
    """
    
    def __init__(self, model_type: str = "classifier", num_classes: int = 20, embedding_dim: int = 768):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Simple linear layer for appearance
        self.dummy_layer = nn.Linear(3 * 224 * 224, num_classes if model_type == "classifier" else embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns dummy predictions.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits (classifier) or embeddings (ssl)
        """
        batch_size = x.shape[0]
        
        if self.model_type == "classifier":
            # Return random logits
            return torch.randn(batch_size, self.num_classes, device=x.device)
        else:  # ssl
            # Return normalized embeddings
            embeddings = torch.randn(batch_size, self.embedding_dim, device=x.device)
            return torch.nn.functional.normalize(embeddings, dim=1)


# Dummy class names for Safari dataset
CLASS_NAMES = [
    "antelope", "bear", "cheetah", "chimpanzee", "crocodile", 
    "elephant", "flamingo", "giraffe", "gorilla", "hippo", 
    "hyena", "leopard", "lion", "meerkat", "ostrich", 
    "rhino", "snake", "tiger", "warthog", "zebra"
]


def load_model():
    """
    Load the inference model based on configuration.
    
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model type: {MODEL_TYPE} on device: {DEVICE}")
    
    if MODEL_TYPE == "dummy":
        # Use dummy model for testing
        model = DummyModel(model_type="classifier", num_classes=len(CLASS_NAMES))
        logger.info("Loaded dummy model (for testing)")
    
    elif MODEL_TYPE == "classifier":
        # In future, load real classifier
        # For now, use dummy
        model = DummyModel(model_type="classifier", num_classes=len(CLASS_NAMES))
        
        if MODEL_CHECKPOINT:
            logger.info(f"Checkpoint specified: {MODEL_CHECKPOINT}")
            logger.warning("Real model loading not implemented yet, using dummy model")
        else:
            logger.info("No checkpoint provided, using dummy model")
    
    elif MODEL_TYPE == "ssl":
        # In future, load SSL model
        # For now, use dummy
        model = DummyModel(model_type="ssl", embedding_dim=768)
        logger.info("Using dummy SSL model")
    
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    model = model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully")
    
    return model


def run_inference(model, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    Run inference on batch of frames.
    
    Args:
        model: PyTorch model
        frames: List of frames (numpy arrays)
    
    Returns:
        List of prediction dictionaries
    """
    # Preprocess
    batch = preprocess_frames(frames).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        if MODEL_TYPE in ["classifier", "dummy"]:
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            results = []
            for i in range(len(frames)):
                class_idx = int(predictions[i].item())
                # Ensure index is within bounds (safety for dummy model)
                class_idx = class_idx % len(CLASS_NAMES)
                
                results.append({
                    "class_id": class_idx,
                    "class_name": CLASS_NAMES[class_idx],
                    "confidence": float(confidences[i].item()),
                    "probabilities": probabilities[i].cpu().tolist()
                })
        
        elif MODEL_TYPE == "ssl":
            embeddings = model(batch)
            
            results = []
            for i in range(len(frames)):
                results.append({
                    "embedding": embeddings[i].cpu().tolist(),
                    "embedding_dim": embeddings.shape[1]
                })
        
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    return results
