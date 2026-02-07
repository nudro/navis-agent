"""
Pydantic schemas for structured data exchange between agents.
"""

from typing import List, Optional, Dict, Tuple, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np


class FrameState(BaseModel):
    """Per-frame data for a track."""
    frame_number: int
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    centroid: Optional[Tuple[float, float]] = Field(None, description="Centroid (x, y)")
    velocity: Optional[Tuple[float, float]] = Field(None, description="Velocity vector (vx, vy) in px/s")
    speed: Optional[float] = Field(None, description="Speed magnitude in px/s")
    acceleration: Optional[Tuple[float, float]] = Field(None, description="Acceleration vector (ax, ay) in px/s²")
    direction: Optional[float] = Field(None, description="Direction angle in degrees")
    confidence: Optional[float] = Field(None, description="Detection confidence")
    
    class Config:
        arbitrary_types_allowed = True


class TrackDiagnostics(BaseModel):
    """Quality indicators for a track."""
    occlusion_suspected: bool = Field(default=False, description="Occlusion detected")
    centroid_jump_rejected: bool = Field(default=False, description="Centroid jump was rejected as outlier")
    id_switch_suspected: bool = Field(default=False, description="Possible ID switch detected")
    confidence: float = Field(default=0.0, description="Average detection confidence")
    frames_missing: int = Field(default=0, description="Number of consecutive frames without detection")


class TrackState(BaseModel):
    """Complete track time series."""
    track_id: int
    frame_states: List[FrameState] = Field(default_factory=list, description="Per-frame states")
    diagnostics: TrackDiagnostics = Field(default_factory=TrackDiagnostics, description="Quality diagnostics")
    first_seen_frame: int = Field(..., description="Frame number when track was first detected")
    last_seen_frame: int = Field(..., description="Frame number when track was last detected")
    
    class Config:
        arbitrary_types_allowed = True


class CVAgentConfig(BaseModel):
    """Configuration for CV Agent processing."""
    conf_threshold: float = Field(default=0.25, description="YOLO confidence threshold")
    iou_threshold: float = Field(default=0.3, description="IoU threshold for matching")
    smoothing_factor: float = Field(default=0.7, description="Velocity smoothing factor")
    max_velocity_change_ratio: float = Field(default=2.0, description="Max velocity change ratio for outlier detection")
    max_pixel_change: float = Field(default=100, description="Max pixel change for centroid validation")
    max_frames_without_detection: int = Field(default=10, description="Max frames before track is considered lost")


class CVAgentRequest(BaseModel):
    """Input for CV Agent."""
    frame: Any = Field(..., description="Video frame (numpy array)")
    fps: float = Field(..., description="Frames per second")
    frame_number: int = Field(..., description="Current frame number")
    existing_tracks: Dict[int, Dict[str, Any]] = Field(default_factory=dict, description="Existing tracks from previous frames")
    config: CVAgentConfig = Field(default_factory=CVAgentConfig, description="Processing configuration")
    
    class Config:
        arbitrary_types_allowed = True


class CVAgentResponse(BaseModel):
    """Output from CV Agent."""
    tracks: List[TrackState] = Field(default_factory=list, description="Updated track states")
    new_track_ids: List[int] = Field(default_factory=list, description="IDs of newly created tracks")
    lost_track_ids: List[int] = Field(default_factory=list, description="IDs of tracks that were lost")
    frame_number: int = Field(..., description="Frame number processed")
    detections_count: int = Field(default=0, description="Number of detections in this frame")
    
    class Config:
        arbitrary_types_allowed = True


class EventSummary(BaseModel):
    """High-level event detected in a track."""
    event_type: str = Field(..., description="Type of event: 'turn_sharpness', 'jerkiness', 'erratic_driving', etc.")
    track_id: int = Field(..., description="Track ID where event occurred")
    frame_number: int = Field(..., description="Frame number where event was detected")
    timestamp: datetime = Field(default_factory=datetime.now, description="When event was detected")
    severity: float = Field(..., description="Severity score (0-1)")
    description: str = Field(default="", description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")


class OrchestratorMetrics(BaseModel):
    """Aggregated metrics per video run."""
    video_path: str = Field(..., description="Path to processed video")
    fps: float = Field(..., description="Video FPS")
    total_frames: int = Field(..., description="Total frames processed")
    tracks_count: int = Field(default=0, description="Total number of tracks")
    events: List[EventSummary] = Field(default_factory=list, description="Detected events")
    processing_time_seconds: float = Field(default=0.0, description="Total processing time")
    average_tracks_per_frame: float = Field(default=0.0, description="Average number of tracks per frame")
    
    class Config:
        arbitrary_types_allowed = True


class ChatMessage(BaseModel):
    """Chat message for agent communication."""
    role: str = Field(..., description="'user' or 'agent' or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatRequest(BaseModel):
    """Request to chat with the agent."""
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context (track IDs, frame range, etc.)")


class ChatResponse(BaseModel):
    """Response from agent chat."""
    response: str = Field(..., description="Agent response")
    success: bool = Field(default=True, description="Whether the request was successful")
    error: Optional[str] = Field(default=None, description="Error message if success=False")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    plot_data: Optional[Dict[str, Any]] = Field(default=None, description="Plot data for visualization (metric, track_id, frames, values)")
