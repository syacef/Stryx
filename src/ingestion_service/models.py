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


# ============================================================================
# RTSP Stream Models
# ============================================================================

class RTSPStreamRequest(BaseModel):
    """Request model for starting an RTSP stream."""
    
    stream_url: str = Field(
        ...,
        description="RTSP stream URL",
        example="rtsp://192.168.1.100:554/stream1"
    )
    stream_id: str = Field(
        ...,
        description="Unique identifier for the stream",
        example="camera_trap_01"
    )
    enable_motion_detection: bool = Field(
        True,
        description="Whether to filter frames using motion detection"
    )


class RTSPStreamResponse(BaseModel):
    """Response model for RTSP stream operations."""
    
    stream_id: str = Field(..., description="Stream identifier")
    stream_url: str = Field(..., description="RTSP stream URL")
    status: str = Field(..., description="Stream status (starting/running/stopped/error)")
    enable_motion_detection: bool = Field(..., description="Motion detection enabled")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    stopped_at: Optional[str] = Field(None, description="Stop timestamp")
    frames_processed: int = Field(0, description="Total frames processed")
    frames_sent: int = Field(0, description="Frames sent to Redis queue")
    error_message: Optional[str] = Field(None, description="Error message if status is error")


class RTSPStopRequest(BaseModel):
    """Request model for stopping an RTSP stream."""
    
    stream_id: str = Field(
        ...,
        description="Stream identifier to stop",
        example="camera_trap_01"
    )
