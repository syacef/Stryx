import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import List, Dict, Any
from config import MODEL_TYPE, MODEL_CHECKPOINT, DEVICE
from preprocessing import preprocess_frames

# Try to import onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    
logger = logging.getLogger(__name__)


class DummyModel(nn.Module):
    """
    Dummy model for testing without real model dependencies.
    Returns random predictions based on model type.
    """
    
    def __init__(self, model_type: str = "classifier", num_classes: int = 91, embedding_dim: int = 768):
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


class ONNXModelWrapper:
    """Wrapper for ONNX Runtime inference."""
    
    def __init__(self, path: str, device: str):
        if ort is None:
            raise ImportError("onnxruntime is not installed. Please install onnxruntime-gpu or onnxruntime.")
            
        self.path = path
        
        # Configure providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        providers = []
        if device == "cuda":
            if "CUDAExecutionProvider" not in available_providers:
                logger.error("CUDAExecutionProvider requested but not available in onnxruntime!")
                # We will try to force it anyway in case get_available_providers is misleading 
                # or just to let ONNX Runtime throw the specific error
            providers.append("CUDAExecutionProvider")
        
        # Fallback
        providers.append("CPUExecutionProvider")
        
        logger.info(f"Loading ONNX model from {path} with providers: {providers}")
        
        try:
            self.session = ort.InferenceSession(path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("ONNX session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Run inference on input tensor."""
        # Convert PyTorch tensor to Numpy
        x_np = x.cpu().numpy()
        
        # Run inference
        return self.session.run([self.output_name], {self.input_name: x_np})[0]


# Load class names with fallback
try:
    if os.path.exists('/app/labels.txt'):
        CLASS_NAMES = [line.strip() for line in open('/app/labels.txt').readlines()]
    else:
        logger.warning("labels.txt not found at /app/labels.txt. Using dummy classes.")
        CLASS_NAMES = [f"class_{i}" for i in range(91)]
except Exception as e:
    logger.error(f"Error loading labels: {e}")
    CLASS_NAMES = [f"class_{i}" for i in range(91)]


def load_model():
    """
    Load the inference model based on configuration.
    
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model type: {MODEL_TYPE} on device: {DEVICE}")
    
    # 1. Check for ONNX checkpoint first
    if MODEL_CHECKPOINT and MODEL_CHECKPOINT.endswith(".onnx"):
        logger.info(f"Detected ONNX checkpoint: {MODEL_CHECKPOINT}")
        if os.path.exists(MODEL_CHECKPOINT):
            try:
                return ONNXModelWrapper(MODEL_CHECKPOINT, DEVICE)
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                # Fallback logic could go here, but for now we let it fall through or error
        else:
            logger.warning(f"ONNX file not found at {MODEL_CHECKPOINT}. Using dummy model.")

    # 2. PyTorch / Dummy Logic
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
        model: PyTorch model or ONNXModelWrapper
        frames: List of frames (numpy arrays)
    
    Returns:
        List of prediction dictionaries
    """
    # Preprocess
    batch = preprocess_frames(frames).to(DEVICE)
    
    # --- ONNX Inference ---
    if isinstance(model, ONNXModelWrapper):
        # Run ONNX inference
        outputs = model(batch)
        
        # Determine if output is classification (probabilities) or embedding
        # For student_tinyvit (SSL), it's embeddings.
        # But if you load a classifier ONNX, it would be logits/probs.
        # We can heuristically check output shape or rely on MODEL_TYPE
        
        if MODEL_TYPE == "ssl":
            results = []
            for i in range(len(frames)):
                results.append({
                    "embedding": outputs[i].tolist(),
                    "embedding_dim": outputs.shape[1]
                })
            return results
            
        else:
            # Assume Classifier (Logits)
            # Apply Softmax if needed. ONNX usually outputs logits.
            # Convert to torch for easy softmax/topk or use numpy
            
            # Using numpy for post-processing
            outputs_tensor = torch.from_numpy(outputs)
            probabilities = torch.softmax(outputs_tensor, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            results = []
            for i in range(len(frames)):
                class_idx = int(predictions[i].item())
                class_idx = class_idx % len(CLASS_NAMES)
                
                results.append({
                    "class_id": class_idx,
                    "class_name": CLASS_NAMES[class_idx],
                    "confidence": float(confidences[i].item()),
                    "probabilities": probabilities[i].tolist()
                })
            return results

    # --- PyTorch Inference ---
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