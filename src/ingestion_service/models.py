"""
Pydantic models for API requests/responses.
"""
from pydantic import BaseModel, Field
from typing import Optional


class VideoRequest(BaseModel):
    """Request model for video processing."""
    
    video_path: str = Field(
        ...,
        description="Relative path to video file within VIDEO_BUCKET",
        example="sample_video.mp4"
    )
    video_id: Optional[str] = Field(
        None,
        description="Optional video identifier (auto-generated if not provided)",
        example="video_001"
    )


class VideoResponse(BaseModel):
    """Response model for video processing request."""
    
    status: str = Field(..., description="Processing status", example="queued")
    video_path: str = Field(..., description="Video path", example="sample_video.mp4")
    video_id: str = Field(..., description="Video identifier", example="video_001")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status", example="healthy")
    redis: dict = Field(..., description="Redis connection info")


class VideoMetadata(BaseModel):
    """Video processing metadata."""
    
    video_id: str
    video_path: str
    status: str
    total_frames: int
    processed_frames: int
    motion_frames: int
    completed_at: str
