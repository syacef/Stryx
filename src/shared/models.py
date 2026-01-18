from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class InferenceResult:
    """Inference result data model."""
    
    video_id: str
    frame_number: int
    timestamp: float
    motion_score: float
    predictions: Dict[str, Any]
    embedding: Optional[List[float]] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "motion_score": self.motion_score,
            "predictions": self.predictions,
            "embedding": self.embedding,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class VideoMetadata:
    """Video processing metadata."""
    
    video_id: str
    video_path: str
    status: str  # "queued", "processing", "completed", "failed"
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    motion_frames: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "status": self.status,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "motion_frames": self.motion_frames,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# Database schema SQL
DB_SCHEMA = """
-- Inference results table
CREATE TABLE IF NOT EXISTS inference_results (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp FLOAT,
    motion_score FLOAT,
    predictions JSONB,
    embedding FLOAT[],
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, frame_number)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_video_id ON inference_results(video_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON inference_results(created_at);
CREATE INDEX IF NOT EXISTS idx_confidence ON inference_results(confidence) WHERE confidence IS NOT NULL;

-- Video metadata table
CREATE TABLE IF NOT EXISTS video_metadata (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(255) UNIQUE NOT NULL,
    video_path TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,
    total_frames INTEGER,
    processed_frames INTEGER,
    motion_frames INTEGER,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_video_metadata_status ON video_metadata(status);
CREATE INDEX IF NOT EXISTS idx_video_metadata_created ON video_metadata(created_at);

-- Create TimescaleDB hypertable for time-series data (optional)
-- SELECT create_hypertable('inference_results', 'created_at', if_not_exists => TRUE);
"""
