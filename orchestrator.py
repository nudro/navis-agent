"""
Orchestrator Agent - Plans steps, routes tool calls, maintains state.
Computes standardized metrics and events, implements fallback logic.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from cv_agent import CVAgent
from motion_core import compute_centroid_distance_px
from schemas import (
    CVAgentRequest,
    CVAgentResponse,
    TrackState,
    FrameState,
    EventSummary,
    OrchestratorMetrics,
    CVAgentConfig,
    ChatRequest,
    ChatResponse,
)


class Orchestrator:
    """Orchestrator Agent for coordinating CV processing and analytics."""
    
    def __init__(self, model_path: str = "models/yolov8n_vessels.pt", config: Optional[CVAgentConfig] = None):
        """Initialize Orchestrator.
        
        Args:
            model_path: Path to YOLO model
            config: CV Agent configuration
        """
        self.cv_agent = CVAgent(model_path=model_path)
        self.config = config or CVAgentConfig()
        
        # Per-video run state
        self.video_path: Optional[str] = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.total_frames: int = 0
        self.processing_start_time: Optional[datetime] = None
        
        # Per-track state
        self.tracks: Dict[int, TrackState] = {}
        self.track_quality: Dict[int, float] = {}  # Track ID -> quality score (0-1)
        
        # Events and metrics
        self.events: List[EventSummary] = []
        self.metrics: Optional[OrchestratorMetrics] = None
    
    def reset(self):
        """Reset orchestrator state for a new video."""
        self.video_path = None
        self.fps = 30.0
        self.frame_count = 0
        self.total_frames = 0
        self.processing_start_time = None
        self.tracks = {}
        self.track_quality = {}
        self.events = []
        self.metrics = None
    
    def sync_tracks_from_processing_state(self, vessel_tracks: Dict[int, Dict[str, Any]], 
                                         frame_count: int, fps: float, video_path: str = None):
        """Sync tracks from existing processing_state format to orchestrator.
        
        Args:
            vessel_tracks: Dict from processing_state["vessel_tracks"]
            frame_count: Current frame count
            fps: Frames per second
            video_path: Optional video path
        """
        self.frame_count = frame_count
        self.fps = fps
        if video_path:
            self.video_path = video_path
        
        # Convert vessel_tracks dict format to TrackState objects
        for track_id, track_data in vessel_tracks.items():
            # Ensure track_data has required fields
            if not track_data.get("frames"):
                continue  # Skip tracks with no frames
            
            track_state = self.cv_agent._dict_to_track_state(track_id, track_data)
            if track_state.frame_states:  # Only add if has frame states
                self.tracks[track_id] = track_state
                
                # Update quality score
                self._update_track_quality(track_id, track_state)
        
        # Remove tracks that are no longer in vessel_tracks
        tracks_to_remove = [tid for tid in self.tracks.keys() if tid not in vessel_tracks]
        for track_id in tracks_to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]
            if track_id in self.track_quality:
                del self.track_quality[track_id]
        
        # Update frame count for event detection (only if we have tracks)
        # Run event detection more frequently to catch events as they happen
        if self.tracks and self.frame_count % 5 == 0:
            self._detect_events()
    
    def run_video(self, video_frames: List[Any], fps: float, video_path: str = "") -> OrchestratorMetrics:
        """Process a video and compute metrics.
        
        Args:
            video_frames: List of video frames (numpy arrays)
            fps: Frames per second
            video_path: Path to video file (optional)
            
        Returns:
            OrchestratorMetrics with aggregated results
        """
        self.reset()
        self.video_path = video_path
        self.fps = fps
        self.total_frames = len(video_frames)
        self.processing_start_time = datetime.now()
        
        # Convert tracks to dict format for CV Agent
        existing_tracks_dict = {}
        
        for frame_idx, frame in enumerate(video_frames):
            self.frame_count = frame_idx + 1
            
            # Convert TrackState list to dict format
            existing_tracks_dict = self.cv_agent.track_states_to_dict(list(self.tracks.values()))
            
            # Create request
            request = CVAgentRequest(
                frame=frame,
                fps=fps,
                frame_number=self.frame_count,
                existing_tracks=existing_tracks_dict,
                config=self.config
            )
            
            # Process frame with CV Agent
            response = self.cv_agent.process_frame(request)
            
            # Update tracks
            for track_state in response.tracks:
                track_id = track_state.track_id
                self.tracks[track_id] = track_state
                
                # Update quality score
                self._update_track_quality(track_id, track_state)
            
            # Remove lost tracks
            for track_id in response.lost_track_ids:
                if track_id in self.tracks:
                    del self.tracks[track_id]
                if track_id in self.track_quality:
                    del self.track_quality[track_id]
            
            # Detect events periodically (every N frames or on significant changes)
            if self.frame_count % 10 == 0:  # Check every 10 frames
                self._detect_events()
            
            # Fallback logic: re-init tracks if quality degrades
            self._apply_fallback_logic()
        
        # Final event detection
        self._detect_events()
        
        # Compute final metrics
        processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        
        self.metrics = OrchestratorMetrics(
            video_path=video_path,
            fps=fps,
            total_frames=self.total_frames,
            tracks_count=len(self.tracks),
            events=self.events,
            processing_time_seconds=processing_time,
            average_tracks_per_frame=len(self.tracks) / max(self.total_frames, 1)
        )
        
        return self.metrics
    
    def _update_track_quality(self, track_id: int, track_state: TrackState):
        """Update quality score for a track."""
        if not track_state.frame_states:
            self.track_quality[track_id] = 0.0
            return
        
        # Quality factors:
        # - Detection confidence
        # - Continuity (no missing frames)
        # - Smoothness (low acceleration variance)
        
        confidences = [fs.confidence for fs in track_state.frame_states if fs.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Continuity score (penalize missing frames)
        continuity = 1.0 - (track_state.diagnostics.frames_missing / max(len(track_state.frame_states), 1))
        continuity = max(0.0, continuity)
        
        # Smoothness score (lower acceleration variance = smoother)
        accelerations = [fs.acceleration for fs in track_state.frame_states if fs.acceleration is not None]
        if accelerations:
            accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in accelerations]
            accel_variance = np.var(accel_magnitudes) if len(accel_magnitudes) > 1 else 0.0
            smoothness = 1.0 / (1.0 + accel_variance)  # Inverse relationship
        else:
            smoothness = 1.0
        
        # Combined quality score
        quality = (avg_confidence * 0.4 + continuity * 0.3 + smoothness * 0.3)
        self.track_quality[track_id] = quality
    
    def _detect_events(self):
        """Detect events in current tracks."""
        for track_id, track_state in self.tracks.items():
            if len(track_state.frame_states) < 3:
                continue
            
            # Detect turn sharpness
            turn_sharpness = self._compute_turn_sharpness(track_state)
            if turn_sharpness > 0.5:  # Threshold for sharp turn
                self._add_event(
                    event_type="turn_sharpness",
                    track_id=track_id,
                    severity=turn_sharpness,
                    description=f"Sharp turn detected (severity: {turn_sharpness:.2f})",
                    metadata={"turn_sharpness": turn_sharpness}
                )
            
            # Detect jerkiness
            jerkiness = self._compute_jerkiness(track_state)
            if jerkiness > 0.5:  # Threshold for jerky motion
                self._add_event(
                    event_type="jerkiness",
                    track_id=track_id,
                    severity=jerkiness,
                    description=f"Jerky motion detected (severity: {jerkiness:.2f})",
                    metadata={"jerkiness": jerkiness}
                )
            
            # Detect erratic driving
            erratic_score = self._compute_erratic_driving(track_state)
            if erratic_score > 0.5:  # Threshold for erratic behavior
                self._add_event(
                    event_type="erratic_driving",
                    track_id=track_id,
                    severity=erratic_score,
                    description=f"Erratic driving detected (severity: {erratic_score:.2f})",
                    metadata={"erratic_score": erratic_score}
                )
    
    def _compute_turn_sharpness(self, track_state: TrackState) -> float:
        """Compute turn sharpness (direction change rate)."""
        if len(track_state.frame_states) < 2:
            return 0.0
        
        directions = [fs.direction for fs in track_state.frame_states if fs.direction is not None]
        if len(directions) < 2:
            return 0.0
        
        # Compute direction changes
        direction_changes = []
        for i in range(1, len(directions)):
            change = abs(directions[i] - directions[i-1])
            # Handle wrap-around (e.g., 350° to 10° = 20° change, not 340°)
            if change > 180:
                change = 360 - change
            direction_changes.append(change)
        
        if not direction_changes:
            return 0.0
        
        # Sharpness = average direction change rate (normalized)
        avg_change = sum(direction_changes) / len(direction_changes)
        sharpness = min(1.0, avg_change / 45.0)  # Normalize to 0-1 (45° = max)
        
        return sharpness
    
    def _compute_jerkiness(self, track_state: TrackState) -> float:
        """Compute jerkiness (acceleration change rate)."""
        if len(track_state.frame_states) < 2:
            return 0.0
        
        accelerations = [fs.acceleration for fs in track_state.frame_states if fs.acceleration is not None]
        if len(accelerations) < 2:
            return 0.0
        
        # Compute acceleration changes (jerk)
        accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in accelerations]
        jerk_values = []
        for i in range(1, len(accel_magnitudes)):
            jerk = abs(accel_magnitudes[i] - accel_magnitudes[i-1])
            jerk_values.append(jerk)
        
        if not jerk_values:
            return 0.0
        
        # Jerkiness = average jerk (normalized)
        avg_jerk = sum(jerk_values) / len(jerk_values)
        jerkiness = min(1.0, avg_jerk / 50.0)  # Normalize to 0-1
        
        return jerkiness
    
    def _compute_erratic_driving(self, track_state: TrackState) -> float:
        """Compute erratic driving score (velocity variance)."""
        if len(track_state.frame_states) < 2:
            return 0.0
        
        velocities = [fs.velocity for fs in track_state.frame_states if fs.velocity is not None]
        if len(velocities) < 2:
            return 0.0
        
        # Compute velocity magnitudes
        speed_values = [math.sqrt(vx**2 + vy**2) for (vx, vy) in velocities]
        
        # Erratic = high variance in speed
        if len(speed_values) < 2:
            return 0.0
        
        speed_variance = np.var(speed_values)
        mean_speed = np.mean(speed_values)
        
        # Coefficient of variation
        if mean_speed > 0:
            cv = math.sqrt(speed_variance) / mean_speed
            erratic_score = min(1.0, cv / 2.0)  # Normalize to 0-1
        else:
            erratic_score = 0.0
        
        return erratic_score
    
    def _add_event(self, event_type: str, track_id: int, severity: float, 
                   description: str = "", metadata: Dict[str, Any] = None):
        """Add an event to the events list (avoid duplicates)."""
        # Check if similar event already exists (same type, track, recent frame)
        for event in self.events:
            if (event.event_type == event_type and 
                event.track_id == track_id and
                abs(event.frame_number - self.frame_count) < 10):
                return  # Skip duplicate
        
        event = EventSummary(
            event_type=event_type,
            track_id=track_id,
            frame_number=self.frame_count,
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            metadata=metadata or {}
        )
        self.events.append(event)
    
    def _apply_fallback_logic(self):
        """Apply fallback logic for degraded tracks."""
        tracks_to_reinit = []
        
        for track_id, quality in self.track_quality.items():
            if quality < 0.3:  # Low quality threshold
                track_state = self.tracks.get(track_id)
                if track_state and track_state.diagnostics.id_switch_suspected:
                    tracks_to_reinit.append(track_id)
        
        # Re-init tracks (for now, just mark for removal - could implement re-init logic)
        for track_id in tracks_to_reinit:
            if track_id in self.tracks:
                # Could implement re-init logic here
                # For now, we'll just remove low-quality tracks
                pass
    
    def query_tracks(self, query: str) -> str:
        """Answer questions about tracked vessels.
        
        Args:
            query: User query string
            
        Returns:
            Response string
        """
        query_lower = query.lower()
        
        # Check for comparison queries first
        if any(word in query_lower for word in ["faster", "slower", "fastest", "slowest", "more", "less", "compare"]):
            # Extract track IDs from query
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                track_ids = [int(n) for n in numbers]
                # Route to comparison if it's a comparison query
                if any(word in query_lower for word in ["faster", "slower", "fastest", "slowest", "more", "less", "compare"]):
                    return self.compare_tracks(track_ids, query)
            elif "which" in query_lower:
                # If asking "which track is faster" without IDs, compare all tracks
                if "faster" in query_lower or "fastest" in query_lower:
                    return self._compare_speed(list(self.tracks.keys()))
                elif "jerkier" in query_lower or "jerky" in query_lower:
                    return self._compare_jerkiness(list(self.tracks.keys()))
                elif "erratic" in query_lower:
                    return self._compare_erratic(list(self.tracks.keys()))
                elif "sharp" in query_lower or "turn" in query_lower:
                    return self._compare_turn_sharpness(list(self.tracks.keys()))
        
        if "how many" in query_lower or "count" in query_lower:
            count = len(self.tracks)
            return f"There are currently {count} tracked vessel(s)."
        
        if "list" in query_lower or "show" in query_lower:
            if not self.tracks:
                return "No vessels are currently being tracked."
            
            track_ids = sorted(self.tracks.keys())
            track_info = []
            for track_id in track_ids[:10]:  # Limit to 10 tracks
                track_state = self.tracks[track_id]
                quality = self.track_quality.get(track_id, 0.0)
                track_info.append(
                    f"Track {track_id}: {len(track_state.frame_states)} frames, "
                    f"quality: {quality:.2f}"
                )
            
            response = "Tracked vessels:\n" + "\n".join(track_info)
            if len(track_ids) > 10:
                response += f"\n... and {len(track_ids) - 10} more tracks."
            return response
        
        if "track" in query_lower and any(char.isdigit() for char in query):
            # Extract track ID from query
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                track_id = int(numbers[0])
                if track_id in self.tracks:
                    # Check if asking for specific metric - prioritize this
                    if any(word in query_lower for word in ["speed", "velocity"]):
                        return self._show_track_speed(track_id)
                    elif "direction" in query_lower:
                        return self._show_track_direction(track_id)
                    elif "acceleration" in query_lower or "accel" in query_lower:
                        return self._show_track_acceleration(track_id)
                    elif "centroid" in query_lower or "position" in query_lower:
                        return self._show_track_centroid(track_id)
                    elif "quality" in query_lower:
                        return self._show_track_quality(track_id)
                    else:
                        # General track info
                        track_state = self.tracks[track_id]
                        quality = self.track_quality.get(track_id, 0.0)
                        last_frame = track_state.frame_states[-1] if track_state.frame_states else None
                        
                        response = f"Track {track_id}:\n"
                        response += f"  Frames: {len(track_state.frame_states)}\n"
                        response += f"  Quality: {quality:.2f}\n"
                        if last_frame:
                            response += f"  Last position: {last_frame.centroid}\n"
                            response += f"  Speed: {last_frame.speed:.2f} px/s\n" if last_frame.speed else "  Speed: N/A\n"
                        return response
                else:
                    return f"Track {track_id} not found."
        
        return "I can help you with information about tracked vessels. Try asking:\n" \
               "- 'How many vessels are tracked?'\n" \
               "- 'List tracked vessels'\n" \
               "- 'Show track 0'\n" \
               "- 'Show speed of track 0'"
    
    def summarize_motion(self) -> str:
        """Generate a summary of all motion calculations from tracked vessels.
        
        Returns:
            Summary string with motion statistics
        """
        if not self.tracks:
            return "No vessels are currently being tracked. Load a video and start tracking to see motion summaries."
        
        # Collect all motion data from all tracks
        all_speeds = []
        all_velocities = []
        all_accelerations = []
        all_directions = []
        track_summaries = []
        
        for track_id, track_state in self.tracks.items():
            if not track_state.frame_states:
                continue
            
            # Extract motion data for this track
            track_speeds = [fs.speed for fs in track_state.frame_states if fs.speed is not None]
            track_velocities = [fs.velocity for fs in track_state.frame_states if fs.velocity is not None]
            track_accelerations = [fs.acceleration for fs in track_state.frame_states if fs.acceleration is not None]
            track_directions = [fs.direction for fs in track_state.frame_states if fs.direction is not None]
            
            if track_speeds:
                all_speeds.extend(track_speeds)
                all_velocities.extend(track_velocities)
                all_directions.extend(track_directions)
            
            if track_accelerations:
                all_accelerations.extend(track_accelerations)
            
            # Per-track summary
            if track_speeds:
                avg_speed = sum(track_speeds) / len(track_speeds)
                max_speed = max(track_speeds)
                min_speed = min(track_speeds)
                
                # Average direction
                if track_directions:
                    # Handle circular direction (0-360 degrees)
                    directions_rad = [math.radians(d) for d in track_directions]
                    avg_sin = sum(math.sin(r) for r in directions_rad) / len(directions_rad)
                    avg_cos = sum(math.cos(r) for r in directions_rad) / len(directions_rad)
                    avg_direction = math.degrees(math.atan2(avg_sin, avg_cos))
                    if avg_direction < 0:
                        avg_direction += 360
                else:
                    avg_direction = None
                
                # Acceleration magnitude stats
                if track_accelerations:
                    accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in track_accelerations]
                    avg_accel = sum(accel_magnitudes) / len(accel_magnitudes)
                    max_accel = max(accel_magnitudes)
                else:
                    avg_accel = None
                    max_accel = None
                
                track_summaries.append({
                    "track_id": track_id,
                    "frames": len(track_state.frame_states),
                    "avg_speed": avg_speed,
                    "max_speed": max_speed,
                    "min_speed": min_speed,
                    "avg_direction": avg_direction,
                    "avg_accel": avg_accel,
                    "max_accel": max_accel
                })
        
        if not all_speeds:
            return "Motion data is not yet available. Wait for more frames to be processed."
        
        # Overall statistics
        overall_avg_speed = sum(all_speeds) / len(all_speeds)
        overall_max_speed = max(all_speeds)
        overall_min_speed = min(all_speeds)
        
        # Average direction (circular mean)
        if all_directions:
            directions_rad = [math.radians(d) for d in all_directions]
            avg_sin = sum(math.sin(r) for r in directions_rad) / len(directions_rad)
            avg_cos = sum(math.cos(r) for r in directions_rad) / len(directions_rad)
            overall_avg_direction = math.degrees(math.atan2(avg_sin, avg_cos))
            if overall_avg_direction < 0:
                overall_avg_direction += 360
        else:
            overall_avg_direction = None
        
        # Acceleration statistics
        if all_accelerations:
            accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in all_accelerations]
            overall_avg_accel = sum(accel_magnitudes) / len(accel_magnitudes)
            overall_max_accel = max(accel_magnitudes)
        else:
            overall_avg_accel = None
            overall_max_accel = None
        
        # Build summary
        summary = f"Motion Summary for {len(self.tracks)} Tracked Vessel(s)\n"
        summary += "=" * 60 + "\n\n"
        
        summary += "OVERALL STATISTICS:\n"
        summary += f"  Total frames analyzed: {sum(len(ts.frame_states) for ts in self.tracks.values())}\n"
        summary += f"  Average speed: {overall_avg_speed:.2f} px/s\n"
        summary += f"  Maximum speed: {overall_max_speed:.2f} px/s\n"
        summary += f"  Minimum speed: {overall_min_speed:.2f} px/s\n"
        
        if overall_avg_direction is not None:
            summary += f"  Average direction: {overall_avg_direction:.1f}°\n"
        
        if overall_avg_accel is not None:
            summary += f"  Average acceleration: {overall_avg_accel:.2f} px/s²\n"
            summary += f"  Maximum acceleration: {overall_max_accel:.2f} px/s²\n"
        
        summary += "\nPER-TRACK BREAKDOWN:\n"
        summary += "-" * 60 + "\n"
        
        for track_sum in track_summaries:
            summary += f"\nTrack {track_sum['track_id']} ({track_sum['frames']} frames):\n"
            summary += f"  Speed: avg={track_sum['avg_speed']:.2f}, max={track_sum['max_speed']:.2f}, min={track_sum['min_speed']:.2f} px/s\n"
            
            if track_sum['avg_direction'] is not None:
                summary += f"  Direction: {track_sum['avg_direction']:.1f}° (average)\n"
            
            if track_sum['avg_accel'] is not None:
                summary += f"  Acceleration: avg={track_sum['avg_accel']:.2f}, max={track_sum['max_accel']:.2f} px/s²\n"
        
        summary += "\n" + "=" * 60
        
        return summary
    
    def summarize_track_motion(self, track_id: int) -> str:
        """Generate detailed summary of motion calculations for a specific track.
        
        Args:
            track_id: Track ID to summarize
            
        Returns:
            Detailed summary string with metrics, values, and time series
        """
        if track_id not in self.tracks:
            return f"Track {track_id} not found. Use 'List tracked vessels' to see available tracks."
        
        track_state = self.tracks[track_id]
        
        if not track_state.frame_states:
            return f"Track {track_id} has no motion data yet."
        
        # Extract all motion data
        frame_numbers = [fs.frame_number for fs in track_state.frame_states]
        centroids = [fs.centroid for fs in track_state.frame_states if fs.centroid]
        velocities = [fs.velocity for fs in track_state.frame_states if fs.velocity is not None]
        speeds = [fs.speed for fs in track_state.frame_states if fs.speed is not None]
        accelerations = [fs.acceleration for fs in track_state.frame_states if fs.acceleration is not None]
        directions = [fs.direction for fs in track_state.frame_states if fs.direction is not None]
        confidences = [fs.confidence for fs in track_state.frame_states if fs.confidence is not None]
        
        # Calculate statistics
        summary = f"Motion Summary for Track {track_id}\n"
        summary += "=" * 70 + "\n\n"
        
        # Basic Info
        summary += "TRACK INFORMATION:\n"
        summary += f"  Track ID: {track_id}\n"
        summary += f"  Total frames: {len(track_state.frame_states)}\n"
        summary += f"  First seen: Frame {track_state.first_seen_frame}\n"
        summary += f"  Last seen: Frame {track_state.last_seen_frame}\n"
        summary += f"  Duration: {track_state.last_seen_frame - track_state.first_seen_frame + 1} frames\n"
        if self.fps > 0:
            duration_seconds = (track_state.last_seen_frame - track_state.first_seen_frame + 1) / self.fps
            summary += f"  Duration: {duration_seconds:.2f} seconds\n"
        summary += f"  Quality score: {self.track_quality.get(track_id, 0.0):.2f}\n"
        summary += f"  Average confidence: {sum(confidences) / len(confidences):.3f}\n" if confidences else "  Average confidence: N/A\n"
        
        # Diagnostics
        summary += "\nDIAGNOSTICS:\n"
        diag = track_state.diagnostics
        summary += f"  Occlusion suspected: {diag.occlusion_suspected}\n"
        summary += f"  Centroid jumps rejected: {diag.centroid_jump_rejected}\n"
        summary += f"  ID switch suspected: {diag.id_switch_suspected}\n"
        summary += f"  Frames missing: {diag.frames_missing}\n"
        
        # Speed Statistics
        if speeds:
            summary += "\nSPEED STATISTICS (px/s):\n"
            summary += f"  Average: {sum(speeds) / len(speeds):.2f}\n"
            summary += f"  Maximum: {max(speeds):.2f}\n"
            summary += f"  Minimum: {min(speeds):.2f}\n"
            summary += f"  Standard deviation: {np.std(speeds):.2f}\n"
        else:
            summary += "\nSPEED STATISTICS: No data available\n"
        
        # Direction Statistics
        if directions:
            summary += "\nDIRECTION STATISTICS (degrees):\n"
            # Circular mean
            directions_rad = [math.radians(d) for d in directions]
            avg_sin = sum(math.sin(r) for r in directions_rad) / len(directions_rad)
            avg_cos = sum(math.cos(r) for r in directions_rad) / len(directions_rad)
            avg_direction = math.degrees(math.atan2(avg_sin, avg_cos))
            if avg_direction < 0:
                avg_direction += 360
            
            summary += f"  Average direction: {avg_direction:.1f}°\n"
            summary += f"  Direction range: {min(directions):.1f}° to {max(directions):.1f}°\n"
            
            # Direction changes
            direction_changes = []
            for i in range(1, len(directions)):
                change = abs(directions[i] - directions[i-1])
                if change > 180:
                    change = 360 - change
                direction_changes.append(change)
            
            if direction_changes:
                summary += f"  Average direction change: {sum(direction_changes) / len(direction_changes):.1f}° per frame\n"
                summary += f"  Maximum direction change: {max(direction_changes):.1f}°\n"
        else:
            summary += "\nDIRECTION STATISTICS: No data available\n"
        
        # Acceleration Statistics
        if accelerations:
            accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in accelerations]
            summary += "\nACCELERATION STATISTICS (px/s²):\n"
            summary += f"  Average magnitude: {sum(accel_magnitudes) / len(accel_magnitudes):.2f}\n"
            summary += f"  Maximum magnitude: {max(accel_magnitudes):.2f}\n"
            summary += f"  Minimum magnitude: {min(accel_magnitudes):.2f}\n"
            summary += f"  Standard deviation: {np.std(accel_magnitudes):.2f}\n"
        else:
            summary += "\nACCELERATION STATISTICS: No data available\n"
        
        # Velocity Statistics
        if velocities:
            velocity_magnitudes = [math.sqrt(vx**2 + vy**2) for (vx, vy) in velocities]
            summary += "\nVELOCITY STATISTICS (px/s):\n"
            summary += f"  Average magnitude: {sum(velocity_magnitudes) / len(velocity_magnitudes):.2f}\n"
            summary += f"  Maximum magnitude: {max(velocity_magnitudes):.2f}\n"
            summary += f"  Minimum magnitude: {min(velocity_magnitudes):.2f}\n"
        else:
            summary += "\nVELOCITY STATISTICS: No data available\n"
        
        # Time Series Data (sample - show every Nth frame to avoid overwhelming output)
        summary += "\nTIME SERIES DATA (sample - every 10th frame):\n"
        summary += "-" * 70 + "\n"
        summary += f"{'Frame':<8} {'Centroid (x,y)':<20} {'Speed':<10} {'Direction':<12} {'Accel Mag':<12}\n"
        summary += "-" * 70 + "\n"
        
        sample_interval = max(1, len(track_state.frame_states) // 20)  # Show ~20 samples
        for i in range(0, len(track_state.frame_states), sample_interval):
            fs = track_state.frame_states[i]
            centroid_str = f"({fs.centroid[0]:.1f},{fs.centroid[1]:.1f})" if fs.centroid else "N/A"
            speed_str = f"{fs.speed:.2f}" if fs.speed is not None else "N/A"
            direction_str = f"{fs.direction:.1f}°" if fs.direction is not None else "N/A"
            
            if fs.acceleration:
                accel_mag = math.sqrt(fs.acceleration[0]**2 + fs.acceleration[1]**2)
                accel_str = f"{accel_mag:.2f}"
            else:
                accel_str = "N/A"
            
            summary += f"{fs.frame_number:<8} {centroid_str:<20} {speed_str:<10} {direction_str:<12} {accel_str:<12}\n"
        
        if len(track_state.frame_states) > sample_interval * 20:
            summary += f"... ({len(track_state.frame_states) - sample_interval * 20} more frames)\n"
        
        # Motion Events for this track
        track_events = [e for e in self.events if e.track_id == track_id]
        if track_events:
            summary += "\nDETECTED EVENTS:\n"
            for event in track_events:
                summary += f"  - {event.event_type} at frame {event.frame_number} (severity: {event.severity:.2f})\n"
                if event.description:
                    summary += f"    {event.description}\n"
        else:
            summary += "\nDETECTED EVENTS: None\n"
        
        summary += "\n" + "=" * 70
        
        return summary
    
    def summarize_multiple_tracks(self, track_ids: List[int]) -> str:
        """Generate summary for multiple tracks.
        
        Args:
            track_ids: List of track IDs to summarize
            
        Returns:
            Combined summary string
        """
        valid_tracks = [tid for tid in track_ids if tid in self.tracks]
        if not valid_tracks:
            return f"None of the specified tracks ({track_ids}) were found. Use 'List tracked vessels' to see available tracks."
        
        summary = f"Motion Summary for {len(valid_tracks)} Track(s): {', '.join(map(str, valid_tracks))}\n"
        summary += "=" * 70 + "\n\n"
        
        # Summarize each track
        for track_id in valid_tracks:
            track_summary = self.summarize_track_motion(track_id)
            # Remove the header line and add track separator
            lines = track_summary.split('\n')
            summary += '\n'.join(lines[1:])  # Skip first line (header)
            summary += "\n\n" + "-" * 70 + "\n\n"
        
        return summary.rstrip()
    
    def compare_tracks(self, track_ids: List[int], query: str) -> str:
        """Compare tracks based on query criteria.
        
        Args:
            track_ids: List of track IDs to compare
            query: Original query string for context
            
        Returns:
            Comparison result string
        """
        valid_tracks = [tid for tid in track_ids if tid in self.tracks]
        if not valid_tracks:
            return f"None of the specified tracks ({track_ids}) were found."
        
        if len(valid_tracks) < 2:
            # If only one track specified, compare with all other tracks
            all_tracks = list(self.tracks.keys())
            if len(all_tracks) >= 2:
                valid_tracks = all_tracks
            else:
                return f"Please specify at least 2 tracks to compare. Found: {valid_tracks}"
        
        query_lower = query.lower()
        comparison = f"Track Comparison: {', '.join(map(str, valid_tracks))}\n"
        comparison += "=" * 70 + "\n\n"
        
        # Force event detection
        if self.tracks:
            self._detect_events()
        
        # Compare based on query - check for specific metrics first
        if "faster" in query_lower or "slower" in query_lower or "fastest" in query_lower or "slowest" in query_lower:
            comparison += self._compare_speed(valid_tracks)
        elif "erratic" in query_lower:
            comparison += self._compare_erratic(valid_tracks)
        elif "jerk" in query_lower or "jerky" in query_lower:
            comparison += self._compare_jerkiness(valid_tracks)
        elif "sharp" in query_lower or "turn" in query_lower:
            comparison += self._compare_turn_sharpness(valid_tracks)
        elif "speed" in query_lower:
            comparison += self._compare_speed(valid_tracks)
        elif "direction" in query_lower:
            comparison += self._compare_direction(valid_tracks)
        else:
            # General comparison - show all metrics
            comparison += self._compare_all_metrics(valid_tracks)
        
        comparison += "\n" + "=" * 70
        return comparison
    
    def _compare_erratic(self, track_ids: List[int]) -> str:
        """Compare tracks by erratic driving score using acceleration and direction standard deviation."""
        results = []
        for track_id in track_ids:
            track_state = self.tracks[track_id]
            
            # Calculate erratic score using acceleration and direction standard deviation
            speeds = [fs.speed for fs in track_state.frame_states if fs.speed is not None]
            accelerations = [fs.acceleration for fs in track_state.frame_states if fs.acceleration is not None]
            directions = [fs.direction for fs in track_state.frame_states if fs.direction is not None]
            
            # Speed variance (coefficient of variation)
            speed_variance_score = 0.0
            if speeds and len(speeds) > 1:
                speed_std = np.std(speeds)
                speed_mean = np.mean(speeds)
                if speed_mean > 0:
                    speed_cv = speed_std / speed_mean
                    speed_variance_score = min(1.0, speed_cv / 2.0)
            
            # Acceleration variance
            accel_variance_score = 0.0
            if accelerations and len(accelerations) > 1:
                accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in accelerations]
                accel_std = np.std(accel_magnitudes)
                accel_mean = np.mean(accel_magnitudes)
                if accel_mean > 0:
                    accel_cv = accel_std / accel_mean
                    accel_variance_score = min(1.0, accel_cv / 2.0)
            
            # Direction change variance
            direction_variance_score = 0.0
            if directions and len(directions) > 1:
                direction_changes = []
                for i in range(1, len(directions)):
                    change = abs(directions[i] - directions[i-1])
                    if change > 180:
                        change = 360 - change
                    direction_changes.append(change)
                
                if direction_changes:
                    dir_change_std = np.std(direction_changes)
                    dir_change_mean = np.mean(direction_changes)
                    if dir_change_mean > 0:
                        dir_change_cv = dir_change_std / dir_change_mean
                        direction_variance_score = min(1.0, dir_change_cv / 2.0)
            
            # Combined erratic score (weighted average)
            erratic_score = (speed_variance_score * 0.4 + accel_variance_score * 0.4 + direction_variance_score * 0.2)
            
            results.append((track_id, erratic_score, speed_variance_score, accel_variance_score, direction_variance_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        output = "ERRATIC DRIVING COMPARISON:\n"
        output += "-" * 70 + "\n"
        output += "Based on speed variance, acceleration variance, and direction change variance:\n\n"
        for track_id, score, speed_var, accel_var, dir_var in results:
            output += f"  Track {track_id}: {score:.3f} (speed_var={speed_var:.3f}, accel_var={accel_var:.3f}, dir_var={dir_var:.3f})\n"
        
        if results:
            output += f"\nMost erratic: Track {results[0][0]} (score: {results[0][1]:.3f})\n"
            if len(results) > 1:
                output += f"Least erratic: Track {results[-1][0]} (score: {results[-1][1]:.3f})\n"
        
        return output
    
    def _compare_jerkiness(self, track_ids: List[int]) -> str:
        """Compare tracks by jerkiness score."""
        results = []
        for track_id in track_ids:
            track_state = self.tracks[track_id]
            jerkiness = self._compute_jerkiness(track_state)
            results.append((track_id, jerkiness))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        output = "JERKINESS COMPARISON:\n"
        output += "-" * 70 + "\n"
        for track_id, score in results:
            output += f"  Track {track_id}: {score:.3f} (higher = more jerky)\n"
        
        output += f"\nMost jerky: Track {results[0][0]} (score: {results[0][1]:.3f})\n"
        output += f"Least jerky: Track {results[-1][0]} (score: {results[-1][1]:.3f})\n"
        
        return output
    
    def _compare_turn_sharpness(self, track_ids: List[int]) -> str:
        """Compare tracks by turn sharpness."""
        results = []
        for track_id in track_ids:
            track_state = self.tracks[track_id]
            sharpness = self._compute_turn_sharpness(track_state)
            results.append((track_id, sharpness))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        output = "TURN SHARPNESS COMPARISON:\n"
        output += "-" * 70 + "\n"
        for track_id, score in results:
            output += f"  Track {track_id}: {score:.3f} (higher = sharper turns)\n"
        
        output += f"\nSharpest turns: Track {results[0][0]} (score: {results[0][1]:.3f})\n"
        output += f"Gentlest turns: Track {results[-1][0]} (score: {results[-1][1]:.3f})\n"
        
        return output
    
    def _compare_speed(self, track_ids: List[int]) -> str:
        """Compare tracks by speed statistics."""
        results = []
        for track_id in track_ids:
            track_state = self.tracks[track_id]
            speeds = [fs.speed for fs in track_state.frame_states if fs.speed is not None]
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                min_speed = min(speeds)
                results.append((track_id, avg_speed, max_speed, min_speed, len(speeds)))
            else:
                results.append((track_id, 0.0, 0.0, 0.0, 0))
        
        # Filter out tracks with no speed data
        results = [r for r in results if r[4] > 0]
        
        if not results:
            return "No speed data available for comparison."
        
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by average speed
        
        output = "SPEED COMPARISON:\n"
        output += "-" * 70 + "\n"
        for track_id, avg_speed, max_speed, min_speed, frame_count in results:
            output += f"  Track {track_id}: avg={avg_speed:.2f} px/s, max={max_speed:.2f} px/s, min={min_speed:.2f} px/s ({frame_count} frames)\n"
        
        if len(results) > 1:
            fastest = results[0]
            slowest = results[-1]
            speed_diff = fastest[1] - slowest[1]
            speed_diff_pct = (speed_diff / slowest[1] * 100) if slowest[1] > 0 else 0
            
            output += f"\nFASTEST: Track {fastest[0]} with average speed of {fastest[1]:.2f} px/s\n"
            output += f"SLOWEST: Track {slowest[0]} with average speed of {slowest[1]:.2f} px/s\n"
            output += f"   Difference: {speed_diff:.2f} px/s ({speed_diff_pct:.1f}% faster)\n"
        else:
            output += f"\nOnly one track available: Track {results[0][0]} (avg speed: {results[0][1]:.2f} px/s)\n"
        
        return output
    
    def _compare_direction(self, track_ids: List[int]) -> str:
        """Compare tracks by direction statistics."""
        output = "DIRECTION COMPARISON:\n"
        output += "-" * 70 + "\n"
        
        for track_id in track_ids:
            track_state = self.tracks[track_id]
            directions = [fs.direction for fs in track_state.frame_states if fs.direction is not None]
            if directions:
                directions_rad = [math.radians(d) for d in directions]
                avg_sin = sum(math.sin(r) for r in directions_rad) / len(directions_rad)
                avg_cos = sum(math.cos(r) for r in directions_rad) / len(directions_rad)
                avg_direction = math.degrees(math.atan2(avg_sin, avg_cos))
                if avg_direction < 0:
                    avg_direction += 360
                
                direction_changes = []
                for i in range(1, len(directions)):
                    change = abs(directions[i] - directions[i-1])
                    if change > 180:
                        change = 360 - change
                    direction_changes.append(change)
                
                avg_change = sum(direction_changes) / len(direction_changes) if direction_changes else 0
                output += f"  Track {track_id}: avg direction={avg_direction:.1f}°, avg change={avg_change:.1f}°/frame\n"
            else:
                output += f"  Track {track_id}: No direction data\n"
        
        return output
    
    def _compare_all_metrics(self, track_ids: List[int]) -> str:
        """Compare tracks across all metrics."""
        output = "COMPREHENSIVE COMPARISON:\n"
        output += "-" * 70 + "\n\n"
        
        # Speed comparison
        output += self._compare_speed(track_ids) + "\n\n"
        
        # Direction comparison
        output += self._compare_direction(track_ids) + "\n\n"
        
        # Erratic comparison
        output += self._compare_erratic(track_ids) + "\n\n"
        
        # Jerkiness comparison
        output += self._compare_jerkiness(track_ids) + "\n\n"
        
        # Turn sharpness comparison
        output += self._compare_turn_sharpness(track_ids)
        
        return output
    
    def _show_track_speed(self, track_id: int) -> str:
        """Show speed information for a specific track."""
        track_state = self.tracks[track_id]
        speeds = [fs.speed for fs in track_state.frame_states if fs.speed is not None]
        
        if not speeds:
            return f"Track {track_id}: No speed data available yet."
        
        response = f"Speed Information for Track {track_id}\n"
        response += "=" * 50 + "\n\n"
        response += f"Total frames with speed data: {len(speeds)}\n\n"
        response += "STATISTICS:\n"
        response += f"  Average speed: {sum(speeds) / len(speeds):.2f} px/s\n"
        response += f"  Maximum speed: {max(speeds):.2f} px/s\n"
        response += f"  Minimum speed: {min(speeds):.2f} px/s\n"
        response += f"  Standard deviation: {np.std(speeds):.2f} px/s\n\n"
        
        # Show recent speeds (last 10 frames)
        recent_speeds = speeds[-10:] if len(speeds) >= 10 else speeds
        recent_frames = [fs.frame_number for fs in track_state.frame_states if fs.speed is not None][-10:]
        
        response += "RECENT SPEEDS (last 10 frames):\n"
        response += f"{'Frame':<10} {'Speed (px/s)':<15}\n"
        response += "-" * 25 + "\n"
        for frame_num, speed in zip(recent_frames[-10:], recent_speeds):
            response += f"{frame_num:<10} {speed:.2f}\n"
        
        return response
    
    def _show_track_direction(self, track_id: int) -> str:
        """Show direction information for a specific track."""
        track_state = self.tracks[track_id]
        directions = [fs.direction for fs in track_state.frame_states if fs.direction is not None]
        
        if not directions:
            return f"Track {track_id}: No direction data available yet."
        
        # Calculate average direction (circular mean)
        directions_rad = [math.radians(d) for d in directions]
        avg_sin = sum(math.sin(r) for r in directions_rad) / len(directions_rad)
        avg_cos = sum(math.cos(r) for r in directions_rad) / len(directions_rad)
        avg_direction = math.degrees(math.atan2(avg_sin, avg_cos))
        if avg_direction < 0:
            avg_direction += 360
        
        # Direction changes
        direction_changes = []
        for i in range(1, len(directions)):
            change = abs(directions[i] - directions[i-1])
            if change > 180:
                change = 360 - change
            direction_changes.append(change)
        
        response = f"Direction Information for Track {track_id}\n"
        response += "=" * 50 + "\n\n"
        response += f"Total frames with direction data: {len(directions)}\n\n"
        response += "STATISTICS:\n"
        response += f"  Average direction: {avg_direction:.1f}°\n"
        response += f"  Direction range: {min(directions):.1f}° to {max(directions):.1f}°\n"
        if direction_changes:
            response += f"  Average direction change: {sum(direction_changes) / len(direction_changes):.1f}° per frame\n"
            response += f"  Maximum direction change: {max(direction_changes):.1f}°\n"
        
        return response
    
    def _show_track_acceleration(self, track_id: int) -> str:
        """Show acceleration information for a specific track."""
        track_state = self.tracks[track_id]
        accelerations = [fs.acceleration for fs in track_state.frame_states if fs.acceleration is not None]
        
        if not accelerations:
            return f"Track {track_id}: No acceleration data available yet."
        
        accel_magnitudes = [math.sqrt(ax**2 + ay**2) for (ax, ay) in accelerations]
        
        response = f"Acceleration Information for Track {track_id}\n"
        response += "=" * 50 + "\n\n"
        response += f"Total frames with acceleration data: {len(accelerations)}\n\n"
        response += "STATISTICS:\n"
        response += f"  Average magnitude: {sum(accel_magnitudes) / len(accel_magnitudes):.2f} px/s²\n"
        response += f"  Maximum magnitude: {max(accel_magnitudes):.2f} px/s²\n"
        response += f"  Minimum magnitude: {min(accel_magnitudes):.2f} px/s²\n"
        response += f"  Standard deviation: {np.std(accel_magnitudes):.2f} px/s²\n"
        
        return response
    
    def _show_track_centroid(self, track_id: int) -> str:
        """Show centroid/position information for a specific track."""
        track_state = self.tracks[track_id]
        centroids = [fs.centroid for fs in track_state.frame_states if fs.centroid]
        
        if not centroids:
            return f"Track {track_id}: No position data available yet."
        
        # Calculate position range
        x_coords = [c[0] for c in centroids]
        y_coords = [c[1] for c in centroids]
        
        response = f"Position (Centroid) Information for Track {track_id}\n"
        response += "=" * 50 + "\n\n"
        response += f"Total frames with position data: {len(centroids)}\n\n"
        response += "STATISTICS:\n"
        response += f"  Average X: {sum(x_coords) / len(x_coords):.2f} pixels\n"
        response += f"  Average Y: {sum(y_coords) / len(y_coords):.2f} pixels\n"
        response += f"  X range: {min(x_coords):.2f} to {max(x_coords):.2f} pixels\n"
        response += f"  Y range: {min(y_coords):.2f} to {max(y_coords):.2f} pixels\n"
        response += f"  Current position: ({centroids[-1][0]:.2f}, {centroids[-1][1]:.2f})\n"
        
        return response
    
    def _show_track_quality(self, track_id: int) -> str:
        """Show quality information for a specific track."""
        track_state = self.tracks[track_id]
        quality = self.track_quality.get(track_id, 0.0)
        diag = track_state.diagnostics
        
        response = f"Quality Information for Track {track_id}\n"
        response += "=" * 50 + "\n\n"
        response += f"Overall quality score: {quality:.3f} (0.0 = poor, 1.0 = excellent)\n\n"
        response += "DIAGNOSTICS:\n"
        response += f"  Occlusion suspected: {diag.occlusion_suspected}\n"
        response += f"  Centroid jumps rejected: {diag.centroid_jump_rejected}\n"
        response += f"  ID switch suspected: {diag.id_switch_suspected}\n"
        response += f"  Frames missing: {diag.frames_missing}\n"
        response += f"  Average confidence: {diag.confidence:.3f}\n"
        
        return response
    
    def _calculate_track_distance(self, track_id1: int, track_id2: int) -> str:
        """Calculate distance between two tracks using their centroids."""
        if track_id1 not in self.tracks:
            return f"Track {track_id1} not found."
        if track_id2 not in self.tracks:
            return f"Track {track_id2} not found."
        
        track1 = self.tracks[track_id1]
        track2 = self.tracks[track_id2]
        
        # Get the most recent centroids from both tracks
        centroid1 = None
        centroid2 = None
        
        if track1.frame_states:
            last_frame1 = track1.frame_states[-1]
            centroid1 = last_frame1.centroid
        
        if track2.frame_states:
            last_frame2 = track2.frame_states[-1]
            centroid2 = last_frame2.centroid
        
        if centroid1 is None:
            return f"Track {track_id1} has no centroid data."
        if centroid2 is None:
            return f"Track {track_id2} has no centroid data."
        
        # Calculate distance
        distance = compute_centroid_distance_px(centroid1, centroid2)
        
        if distance is None:
            return f"Could not calculate distance between Track {track_id1} and Track {track_id2}."
        
        response = f"Distance between Track {track_id1} and Track {track_id2}\n"
        response += "=" * 50 + "\n\n"
        response += f"Track {track_id1} centroid: ({centroid1[0]:.1f}, {centroid1[1]:.1f})\n"
        response += f"Track {track_id2} centroid: ({centroid2[0]:.1f}, {centroid2[1]:.1f})\n"
        response += f"\nCurrent distance: {distance:.2f} pixels\n"
        
        # Also calculate average distance over all frames where both tracks exist
        distances_over_time = []
        frame_numbers = []
        
        # Find common frames
        track1_frames = {fs.frame_number: fs for fs in track1.frame_states if fs.centroid is not None}
        track2_frames = {fs.frame_number: fs for fs in track2.frame_states if fs.centroid is not None}
        
        common_frames = set(track1_frames.keys()) & set(track2_frames.keys())
        
        if common_frames:
            for frame_num in sorted(common_frames):
                c1 = track1_frames[frame_num].centroid
                c2 = track2_frames[frame_num].centroid
                if c1 and c2:
                    dist = compute_centroid_distance_px(c1, c2)
                    if dist is not None:
                        distances_over_time.append(dist)
                        frame_numbers.append(frame_num)
        
        if distances_over_time:
            avg_distance = sum(distances_over_time) / len(distances_over_time)
            min_distance = min(distances_over_time)
            max_distance = max(distances_over_time)
            
            response += f"\nDistance Statistics (over {len(distances_over_time)} common frames):\n"
            response += f"  Average distance: {avg_distance:.2f} pixels\n"
            response += f"  Minimum distance: {min_distance:.2f} pixels\n"
            response += f"  Maximum distance: {max_distance:.2f} pixels\n"
            response += f"  Standard deviation: {np.std(distances_over_time):.2f} pixels\n"
        
        return response
    
    def _generate_plot_data(self, track_id: int, metric_type: str) -> Optional[Dict[str, Any]]:
        """Generate plot data for a specific track and metric.
        
        Args:
            track_id: Track ID
            metric_type: Type of metric ('speed', 'direction', 'acceleration', etc.)
            
        Returns:
            Dictionary with plot data or None if track/metric not available
        """
        if track_id not in self.tracks:
            return None
        
        track_state = self.tracks[track_id]
        
        # Extract data based on metric type
        frames = []
        values = []
        
        if metric_type.lower() in ["speed", "velocity"]:
            for fs in track_state.frame_states:
                if fs.speed is not None:
                    frames.append(fs.frame_number)
                    values.append(fs.speed)
            label = "Speed (px/s)"
            title = f"Speed over Time - Track {track_id}"
            
        elif metric_type.lower() == "direction":
            for fs in track_state.frame_states:
                if fs.direction is not None:
                    frames.append(fs.frame_number)
                    values.append(fs.direction)
            label = "Direction (degrees)"
            title = f"Direction over Time - Track {track_id}"
            
        elif metric_type.lower() in ["acceleration", "accel"]:
            for fs in track_state.frame_states:
                if fs.acceleration is not None:
                    frames.append(fs.frame_number)
                    # Use magnitude of acceleration vector
                    accel_mag = math.sqrt(fs.acceleration[0]**2 + fs.acceleration[1]**2)
                    values.append(accel_mag)
            label = "Acceleration Magnitude (px/s²)"
            title = f"Acceleration over Time - Track {track_id}"
            
        else:
            return None
        
        if not frames or not values:
            return None
        
        return {
            "metric": metric_type.lower(),
            "track_id": track_id,
            "frames": frames,
            "values": values,
            "label": label,
            "title": title
        }
    
    def query_metrics(self, query: str) -> str:
        """Answer questions about metrics and events.
        
        Args:
            query: User query string
            
        Returns:
            Response string
        """
        query_lower = query.lower()
        
        # Force event detection before querying if we have tracks
        if self.tracks:
            self._detect_events()
        
        if "event" in query_lower or "incident" in query_lower:
            if not self.events:
                return "No events detected yet. Events are detected based on motion patterns (sharp turns, jerkiness, erratic driving)."
            
            response = f"Found {len(self.events)} event(s):\n"
            for event in self.events[-10:]:  # Last 10 events
                response += f"  - {event.event_type} (Track {event.track_id}, Frame {event.frame_number}, " \
                           f"Severity: {event.severity:.2f})\n"
            return response
        
        if "metric" in query_lower or "statistic" in query_lower:
            if not self.metrics:
                return "No metrics available yet. Process a video first."
            
            response = f"Video Metrics:\n"
            response += f"  Total frames: {self.metrics.total_frames}\n"
            response += f"  Tracks: {self.metrics.tracks_count}\n"
            response += f"  Events: {len(self.metrics.events)}\n"
            response += f"  Processing time: {self.metrics.processing_time_seconds:.2f}s\n"
            response += f"  Avg tracks/frame: {self.metrics.average_tracks_per_frame:.2f}\n"
            return response
        
        if "sharp" in query_lower or "turn" in query_lower:
            turn_events = [e for e in self.events if e.event_type == "turn_sharpness"]
            if turn_events:
                response = f"Found {len(turn_events)} sharp turn event(s):\n"
                for event in turn_events[-5:]:
                    response += f"  - Track {event.track_id} at frame {event.frame_number}\n"
                return response
            return "No sharp turns detected."
        
        if "jerk" in query_lower:
            jerk_events = [e for e in self.events if e.event_type == "jerkiness"]
            if jerk_events:
                response = f"Found {len(jerk_events)} jerkiness event(s):\n"
                for event in jerk_events[-5:]:
                    response += f"  - Track {event.track_id} at frame {event.frame_number}\n"
                return response
            return "No jerky motion detected."
        
        if "erratic" in query_lower:
            # Calculate erratic scores for all tracks using standard deviation
            if not self.tracks:
                return "No tracks available for comparison."
            
            # Compare all tracks for erratic behavior
            track_ids = list(self.tracks.keys())
            if len(track_ids) < 2:
                return f"Need at least 2 tracks to compare erratic behavior. Found {len(track_ids)} track(s)."
            
            return self._compare_erratic(track_ids)
        
        return "I can help you with metrics and events. Try asking:\n" \
               "- 'Show events'\n" \
               "- 'Show metrics'\n" \
               "- 'Show sharp turns'\n" \
               "- 'Show jerky motion'"
    
    def execute_command(self, command: str) -> str:
        """Execute control commands.
        
        Args:
            command: Command string
            
        Returns:
            Response string
        """
        command_lower = command.lower().strip()
        
        if "reset" in command_lower:
            self.reset()
            return "Orchestrator state reset."
        
        if "status" in command_lower:
            response = f"Orchestrator Status:\n"
            response += f"  Video: {self.video_path or 'None'}\n"
            response += f"  FPS: {self.fps}\n"
            response += f"  Frame: {self.frame_count}/{self.total_frames}\n"
            response += f"  Tracks: {len(self.tracks)}\n"
            response += f"  Events: {len(self.events)}\n"
            return response
        
        return f"Unknown command: {command}. Available commands: 'reset', 'status'"
    
    def handle_chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Handle chat request and route to appropriate method.
        
        Args:
            chat_request: Chat request
            
        Returns:
            Chat response
        """
        message = chat_request.message.lower()
        
        # Check for help/capabilities queries first
        if any(word in message for word in ["what tools", "what can you", "help", "capabilities", "what do you", "what are you", "who are you", "what is navis"]):
            response_text = "Welcome to NAVIS (Nautical Analysis & Vessel Intelligence System)\n\n" \
                           "I'm your intelligent maritime analysis assistant. Here's what I can do:\n\n" \
                           "TRACKING & VESSEL INFORMATION:\n" \
                           "  • List all tracked vessels\n" \
                           "  • Show detailed information for specific tracks\n" \
                           "  • Query vessel metrics (speed, direction, acceleration, position, quality)\n" \
                           "  • Calculate distances between vessels\n\n" \
                           "MOTION ANALYSIS:\n" \
                           "  • Summarize motion for individual or multiple tracks\n" \
                           "  • Compare vessels (speed, erratic behavior, jerkiness, turn sharpness)\n" \
                           "  • Detect and report events (sharp turns, jerky motion, erratic driving)\n" \
                           "  • Generate interactive plots (speed, direction, acceleration over time)\n\n" \
                           "VISUALIZATION:\n" \
                           "  • Create plots: 'Draw me a plot of speed for track 1'\n" \
                           "  • Graph direction, acceleration, or other metrics\n\n" \
                           "SYSTEM COMMANDS:\n" \
                           "  • Check system status\n" \
                           "  • Reset tracking state\n\n" \
                           "Try asking me:\n" \
                           "  • 'List tracked vessels'\n" \
                           "  • 'Show speed of track 0'\n" \
                           "  • 'Summarize motion for track 1'\n" \
                           "  • 'Which vessel was more erratic?'\n" \
                           "  • 'Draw me a plot of speed for track 1'"
            return ChatResponse(
                response=response_text,
                success=True,
                metadata={"handler": "help"}
            )
        
        # Route to appropriate handler - check event/metric keywords FIRST (more specific)
        # Check for plot requests FIRST (highest priority for visualization)
        if any(word in message for word in ["plot", "draw", "graph", "chart"]) and "track" in message:
            # Check for plot requests
            import re
            numbers = re.findall(r'-?\d+', message)  # Allow negative numbers
            metric_type = None
            
            # Determine metric type
            if any(word in message for word in ["speed", "velocity"]):
                metric_type = "speed"
            elif "direction" in message:
                metric_type = "direction"
            elif any(word in message for word in ["acceleration", "accel"]):
                metric_type = "acceleration"
            else:
                # Default to speed if not specified
                metric_type = "speed"
            
            if numbers:
                track_id = int(numbers[0])
                plot_data = self._generate_plot_data(track_id, metric_type)
                if plot_data:
                    response_text = f"Generating {metric_type} plot for Track {track_id}..."
                    # Return response with plot_data
                    return ChatResponse(
                        response=response_text,
                        success=True,
                        metadata={"handler": "plot"},
                        plot_data=plot_data
                    )
                else:
                    response_text = f"Could not generate plot for Track {track_id}. Track may not exist or have no {metric_type} data."
                    return ChatResponse(
                        response=response_text,
                        success=False,
                        metadata={"handler": "plot"}
                    )
            else:
                response_text = "Please specify a track ID. Example: 'draw me a plot of speed for track 1'"
                return ChatResponse(
                    response=response_text,
                    success=False,
                    metadata={"handler": "plot"}
                )
        # Check for track-specific summary first (single or multiple tracks)
        elif any(word in message for word in ["summary", "summarize"]) and ("track" in message or "track-id" in message or "vessel" in message):
            # Extract all track IDs from message
            import re
            numbers = re.findall(r'\d+', message)
            if numbers:
                track_ids = [int(n) for n in numbers]
                # Check if asking for comparison
                if any(word in message for word in ["more", "less", "compare", "which", "better", "worse", "erratic", "jerky", "sharp"]):
                    response_text = self.compare_tracks(track_ids, message)
                else:
                    response_text = self.summarize_multiple_tracks(track_ids)
            else:
                response_text = "Please specify track ID(s). Example: 'summarize motion for track 1' or 'summarize motion for track 1 and track 2'"
        elif any(word in message for word in ["summary", "summarize", "motion summary"]):
            response_text = self.summarize_motion()
        elif any(word in message for word in ["metric", "event", "statistic", "sharp", "jerk", "erratic", "incident"]):
            response_text = self.query_metrics(chat_request.message)
        elif any(word in message for word in ["reset", "status", "command"]):
            response_text = self.execute_command(chat_request.message)
        elif any(word in message for word in ["distance", "between"]) and "track" in message:
            # Check for distance queries between tracks
            import re
            numbers = re.findall(r'\d+', message)
            if len(numbers) >= 2:
                track_id1 = int(numbers[0])
                track_id2 = int(numbers[1])
                response_text = self._calculate_track_distance(track_id1, track_id2)
            elif len(numbers) == 1:
                response_text = "Please specify two track IDs. Example: 'what is the distance between track 1 and track 7'"
            else:
                response_text = "Please specify two track IDs. Example: 'what is the distance between track 1 and track 7'"
        elif any(word in message for word in ["show"]) and any(word in message for word in ["speed", "velocity", "direction", "acceleration", "accel", "centroid", "position", "quality"]):
            # Check for "show [metric] of track X" queries first - highest priority
            import re
            numbers = re.findall(r'\d+', message)
            if numbers:
                track_id = int(numbers[0])
                if track_id in self.tracks:
                    if any(word in message for word in ["speed", "velocity"]):
                        response_text = self._show_track_speed(track_id)
                    elif "direction" in message:
                        response_text = self._show_track_direction(track_id)
                    elif "acceleration" in message or "accel" in message:
                        response_text = self._show_track_acceleration(track_id)
                    elif "centroid" in message or "position" in message:
                        response_text = self._show_track_centroid(track_id)
                    elif "quality" in message:
                        response_text = self._show_track_quality(track_id)
                    else:
                        response_text = self.query_tracks(chat_request.message)
                else:
                    response_text = f"Track {track_id} not found."
            else:
                response_text = "Please specify a track ID. Example: 'show speed of track 7'"
        elif any(word in message for word in ["track", "vessel", "how many", "list", "which", "faster", "slower", "jerkier", "erratic"]):
            # Check for comparison queries with track IDs
            import re
            numbers = re.findall(r'\d+', message)
            if numbers and any(word in message for word in ["faster", "slower", "jerkier", "erratic", "compare", "more", "less"]):
                track_ids = [int(n) for n in numbers]
                response_text = self.compare_tracks(track_ids, message)
            elif not any(word in message for word in ["event", "metric", "statistic"]):
                response_text = self.query_tracks(chat_request.message)
            else:
                # If it mentions both, prioritize metrics
                response_text = self.query_metrics(chat_request.message)
        elif "show" in message:
            # "show" is ambiguous - check context
            if any(word in message for word in ["event", "metric", "statistic", "sharp", "jerk", "erratic"]):
                response_text = self.query_metrics(chat_request.message)
            elif any(word in message for word in ["track", "vessel"]):
                response_text = self.query_tracks(chat_request.message)
            else:
                # Default: try tracks first, then metrics
                response_text = self.query_tracks(chat_request.message)
                if "I can help" in response_text:
                    response_text = self.query_metrics(chat_request.message)
        else:
            # Default: try all handlers
            response_text = self.query_tracks(chat_request.message)
            if "I can help" in response_text:
                response_text = self.query_metrics(chat_request.message)
                if "I can help" in response_text:
                    response_text = "Welcome to NAVIS (Nautical Analysis & Vessel Intelligence System)\n\n" \
                                   "I'm your intelligent maritime analysis assistant. Here's what I can do:\n\n" \
                                   "TRACKING & VESSEL INFORMATION:\n" \
                                   "  • List all tracked vessels\n" \
                                   "  • Show detailed information for specific tracks\n" \
                                   "  • Query vessel metrics (speed, direction, acceleration, position, quality)\n" \
                                   "  • Calculate distances between vessels\n\n" \
                                   "MOTION ANALYSIS:\n" \
                                   "  • Summarize motion for individual or multiple tracks\n" \
                                   "  • Compare vessels (speed, erratic behavior, jerkiness, turn sharpness)\n" \
                                   "  • Detect and report events (sharp turns, jerky motion, erratic driving)\n" \
                                   "  • Generate interactive plots (speed, direction, acceleration over time)\n\n" \
                                   "VISUALIZATION:\n" \
                                   "  • Create plots: 'Draw me a plot of speed for track 1'\n" \
                                   "  • Graph direction, acceleration, or other metrics\n\n" \
                                   "SYSTEM COMMANDS:\n" \
                                   "  • Check system status\n" \
                                   "  • Reset tracking state\n\n" \
                                   "Try asking me:\n" \
                                   "  • 'List tracked vessels'\n" \
                                   "  • 'Show speed of track 0'\n" \
                                   "  • 'Summarize motion for track 1'\n" \
                                   "  • 'Which vessel was more erratic?'\n" \
                                   "  • 'Draw me a plot of speed for track 1'"
        
        return ChatResponse(
            response=response_text,
            success=True,
            metadata={"handler": "orchestrator"}
        )

