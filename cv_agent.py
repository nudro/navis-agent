"""
CV Agent - Processes video frames using motion core functions.
Produces structured track time series with diagnostics.
"""

import math
from typing import Dict, List, Optional, Any
from pathlib import Path
from ultralytics import YOLO
import torch

from motion_core import (
    detect_vessels_with_yolo,
    match_detections_to_tracks,
    compute_bbox_centroid,
    compute_velocity,
    compute_speed,
    compute_acceleration,
    compute_direction,
    smooth_velocity,
    validate_centroid_change,
)
from schemas import (
    CVAgentRequest,
    CVAgentResponse,
    TrackState,
    FrameState,
    TrackDiagnostics,
    CVAgentConfig,
)


class CVAgent:
    """Computer Vision Agent for vessel detection and tracking."""
    
    def __init__(self, model_path: str = "models/yolov8n_vessels.pt"):
        """Initialize CV Agent with YOLO model.
        
        Args:
            model_path: Path to YOLO model file (default: models/yolov8n_vessels.pt)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        
        self.yolo_model = YOLO(str(self.model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model.to(device)
        self.device = device
    
    def process_frame(self, request: CVAgentRequest) -> CVAgentResponse:
        """Process a single frame and update tracks.
        
        Args:
            request: CVAgentRequest with frame, fps, existing_tracks, and config
            
        Returns:
            CVAgentResponse with updated tracks and diagnostics
        """
        frame = request.frame
        fps = request.fps
        frame_number = request.frame_number
        existing_tracks_dict = request.existing_tracks
        config = request.config
        
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        # Step 1: Detect vessels using YOLO
        detections = detect_vessels_with_yolo(
            frame, 
            self.yolo_model, 
            conf_threshold=config.conf_threshold
        )
        
        # Step 2: Match detections to existing tracks
        matches, unmatched_detections = match_detections_to_tracks(
            detections,
            existing_tracks_dict,
            iou_threshold=config.iou_threshold
        )
        
        # Step 3: Convert existing tracks dict to TrackState objects
        track_states: Dict[int, TrackState] = {}
        for track_id, track_data in existing_tracks_dict.items():
            track_states[track_id] = self._dict_to_track_state(track_id, track_data)
        
        # Step 4: Update matched tracks
        updated_track_ids = set()
        for track_id, det_idx in matches.items():
            if track_id not in track_states:
                continue
            
            detection = detections[det_idx]
            bbox = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 0.0
            centroid = compute_bbox_centroid(bbox)
            
            track_state = track_states[track_id]
            
            # Validate centroid change
            prev_centroid = None
            if track_state.frame_states:
                prev_centroid = track_state.frame_states[-1].centroid
            
            centroid_valid = validate_centroid_change(
                prev_centroid,
                centroid,
                max_pixel_change=config.max_pixel_change
            )
            
            # Compute kinematics if we have previous state
            velocity = None
            speed = None
            direction = None
            acceleration = None
            
            if prev_centroid and centroid_valid:
                # Compute velocity
                raw_velocity = compute_velocity(prev_centroid, centroid, dt)
                
                # Smooth velocity
                prev_velocities = [
                    fs.velocity for fs in track_state.frame_states 
                    if fs.velocity is not None
                ]
                velocity = smooth_velocity(
                    raw_velocity,
                    prev_velocities,
                    smoothing_factor=config.smoothing_factor,
                    max_change_ratio=config.max_velocity_change_ratio
                )
                
                speed = compute_speed(velocity)
                direction = compute_direction(velocity)
                
                # Compute acceleration if we have previous velocity
                if len(track_state.frame_states) > 0 and track_state.frame_states[-1].velocity:
                    prev_velocity = track_state.frame_states[-1].velocity
                    acceleration = compute_acceleration(prev_velocity, velocity, dt)
            
            # Create frame state
            frame_state = FrameState(
                frame_number=frame_number,
                bbox=bbox,
                centroid=centroid,
                velocity=velocity,
                speed=speed,
                acceleration=acceleration,
                direction=direction,
                confidence=confidence
            )
            
            # Update diagnostics
            diagnostics = track_state.diagnostics
            if not centroid_valid:
                diagnostics.centroid_jump_rejected = True
            
            # Update track state
            track_state.frame_states.append(frame_state)
            track_state.last_seen_frame = frame_number
            track_state.diagnostics = diagnostics
            
            updated_track_ids.add(track_id)
        
        # Step 5: Create new tracks for unmatched detections
        new_track_ids = []
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            bbox = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 0.0
            centroid = compute_bbox_centroid(bbox)
            
            # Find next available track ID
            track_id = max(track_states.keys()) + 1 if track_states else 0
            
            frame_state = FrameState(
                frame_number=frame_number,
                bbox=bbox,
                centroid=centroid,
                velocity=None,
                speed=None,
                acceleration=None,
                direction=None,
                confidence=confidence
            )
            
            track_state = TrackState(
                track_id=track_id,
                frame_states=[frame_state],
                diagnostics=TrackDiagnostics(confidence=confidence),
                first_seen_frame=frame_number,
                last_seen_frame=frame_number
            )
            
            track_states[track_id] = track_state
            new_track_ids.append(track_id)
        
        # Step 6: Check for lost tracks (tracks not matched)
        lost_track_ids = []
        for track_id in track_states.keys():
            if track_id not in updated_track_ids and track_id not in new_track_ids:
                track_state = track_states[track_id]
                track_state.diagnostics.frames_missing += 1
                
                # Mark as lost if exceeds threshold
                if track_state.diagnostics.frames_missing > config.max_frames_without_detection:
                    lost_track_ids.append(track_id)
        
        # Step 7: Detect ID switches and occlusions
        for track_id, track_state in track_states.items():
            if len(track_state.frame_states) < 2:
                continue
            
            # Check for ID switch suspicion (large IoU drop with another track)
            # This is a simplified check - could be enhanced
            if track_state.diagnostics.id_switch_suspected:
                continue
            
            # Check for occlusion (missing detection but track continues)
            if track_state.diagnostics.frames_missing > 0:
                track_state.diagnostics.occlusion_suspected = True
        
        # Convert back to list
        tracks_list = list(track_states.values())
        
        return CVAgentResponse(
            tracks=tracks_list,
            new_track_ids=new_track_ids,
            lost_track_ids=lost_track_ids,
            frame_number=frame_number,
            detections_count=len(detections)
        )
    
    def _dict_to_track_state(self, track_id: int, track_data: Dict[str, Any]) -> TrackState:
        """Convert existing track dict format to TrackState."""
        frame_states = []
        
        centroids = track_data.get("centroids", [])
        bboxes = track_data.get("bboxes", [])
        velocities = track_data.get("velocities", [])
        speeds = track_data.get("speeds", [])
        accelerations = track_data.get("accelerations", [])
        directions = track_data.get("directions", [])
        frames = track_data.get("frames", [])
        confidences = track_data.get("confidences", [])
        
        # Create FrameState for each frame
        for i in range(len(frames)):
            frame_num = frames[i] if i < len(frames) else i
            bbox = bboxes[i] if i < len(bboxes) else None
            centroid = centroids[i] if i < len(centroids) else None
            velocity = velocities[i] if i < len(velocities) else None
            speed = speeds[i] if i < len(speeds) else None
            acceleration = accelerations[i] if i < len(accelerations) else None
            direction = directions[i] if i < len(directions) else None
            confidence = confidences[i] if i < len(confidences) else None
            
            if bbox is None:
                continue
            
            frame_state = FrameState(
                frame_number=frame_num,
                bbox=bbox[:4] if isinstance(bbox, list) and len(bbox) >= 4 else [0, 0, 0, 0],
                centroid=centroid if isinstance(centroid, tuple) else None,
                velocity=velocity if isinstance(velocity, tuple) else None,
                speed=speed,
                acceleration=acceleration if isinstance(acceleration, tuple) else None,
                direction=direction,
                confidence=confidence
            )
            frame_states.append(frame_state)
        
        # Extract diagnostics
        diagnostics = TrackDiagnostics()
        if confidences:
            diagnostics.confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        first_frame = frames[0] if frames else 0
        last_frame = frames[-1] if frames else 0
        
        return TrackState(
            track_id=track_id,
            frame_states=frame_states,
            diagnostics=diagnostics,
            first_seen_frame=first_frame,
            last_seen_frame=last_frame
        )
    
    def track_states_to_dict(self, track_states: List[TrackState]) -> Dict[int, Dict[str, Any]]:
        """Convert TrackState list back to dict format for compatibility."""
        result = {}
        
        for track_state in track_states:
            track_data = {
                "centroids": [fs.centroid for fs in track_state.frame_states if fs.centroid],
                "velocities": [fs.velocity for fs in track_state.frame_states if fs.velocity],
                "speeds": [fs.speed for fs in track_state.frame_states if fs.speed],
                "accelerations": [fs.acceleration for fs in track_state.frame_states if fs.acceleration],
                "directions": [fs.direction for fs in track_state.frame_states if fs.direction],
                "bboxes": [fs.bbox for fs in track_state.frame_states],
                "frames": [fs.frame_number for fs in track_state.frame_states],
                "confidences": [fs.confidence for fs in track_state.frame_states if fs.confidence],
            }
            result[track_state.track_id] = track_data
        
        return result
