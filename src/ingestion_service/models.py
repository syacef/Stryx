from pydantic import BaseModel, Field, field_validator
from typing import Optional


class StreamRegisterRequest(BaseModel):
    """Request model for stream registration."""
    rtsp_url: str = Field(
        ...,
        description="RTSP stream URL",
        example="rtsp://camera1.example.com:554/stream"
    )
    stream_id: Optional[str] = Field(
        None,
        description="Optional stream identifier (auto-generated if not provided)",
        example="camera_front_entrance"
    )
    name: Optional[str] = Field(
        None,
        description="Human-readable stream name",
        example="Front Entrance Camera"
    )
    fps: Optional[int] = Field(
        5,
        description="Target frames per second to extract",
        ge=1,
        le=30,
        example=5
    )
    resolution: Optional[tuple[int, int]] = Field(
        None,
        description="Target resolution (width, height). None to keep original.",
        example=(1280, 720)
    )


class StreamResponse(BaseModel):
    """Response model for stream registration."""
    stream_id: str = Field(..., description="Stream identifier")
    rtsp_url: str = Field(..., description="RTSP URL")
    name: Optional[str] = Field(None, description="Stream name")
    status: str = Field(..., description="Stream status", example="registered")
    worker_id: str = Field(..., description="Assigned worker ID")


class StreamDeleteResponse(BaseModel):
    """Response model for stream deletion."""
    stream_id: str = Field(..., description="Stream identifier")
    status: str = Field(..., description="Deletion status", example="deleted")
    message: str = Field(..., description="Deletion message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status", example="healthy")
    redis: dict = Field(..., description="Redis connection and service info")


class WorkerStatusResponse(BaseModel):
    """Response model for worker status."""
    worker_id: str = Field(..., description="Worker identifier")
    status: str = Field(..., description="Worker status")
    assigned_streams: list[str] = Field(..., description="List of assigned stream IDs")
    stream_count: int = Field(..., description="Number of assigned streams")
    last_heartbeat: Optional[str] = Field(None, description="Last heartbeat timestamp")


class FrameMessage(BaseModel):
    """Frame message for Redis queue."""
    stream_id: str = Field(..., description="Stream identifier")
    frame_id: str = Field(..., description="Unique frame identifier")
    timestamp: float = Field(..., description="Frame timestamp in seconds")
    frame_number: int = Field(..., description="Sequential frame number")
    frame_data: str = Field(..., description="Base64 or hex encoded frame data")
    shape: tuple[int, int] = Field(..., description="Frame shape (height, width)")
    worker_id: str = Field(..., description="Worker that processed this frame")
