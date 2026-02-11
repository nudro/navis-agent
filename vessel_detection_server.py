"""
Vessel Detection Server - FastAPI Backend
Provides WebSocket streaming for video frames and detection data.
"""

import cv2
import numpy as np
import json
import math
import asyncio
import base64
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import torch
import glob
from datetime import datetime

# Import motion core functions and agents
from motion_core import (
    compute_bbox_centroid,
    compute_bbox_area,
    detect_vessels_with_yolo,
    compute_velocity,
    compute_speed,
    compute_acceleration,
    compute_direction,
    smooth_velocity,
    validate_centroid_change,
    compute_iou,
    match_detections_to_tracks,
    compute_centroid_distance_px,
    compute_bbox_edge_distance_px,
)
from orchestrator import Orchestrator
from schemas import ChatRequest, ChatResponse
from c3_rest_wrapper import get_c3_wrapper

app = FastAPI(title="NAVIS - Vessel Detection Server")

# Mount static files (frontend) - only if dist directory exists
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# CORS middleware - allow all origins in container (C3.ai integration)
# In production, restrict to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for C3.ai integration
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
video_cap: Optional[cv2.VideoCapture] = None
yolo_model: Optional[YOLO] = None
active_connections: Set[WebSocket] = set()
chat_connections: Set[WebSocket] = set()
orchestrator: Optional[Orchestrator] = None
processing_state = {
    "video_path": None,
    "current_frame": None,
    "frame_count": 0,
    "fps": 30,
    "width": 0,
    "height": 0,
    "is_paused": True,
    "is_tracking": False,
    "selected_track_ids": set(),
    "vessel_tracks": {},  # {track_id: {centroids, velocities, speeds, accelerations, directions, bboxes, frames}}
    "active_track_ids": set(),
    "next_track_id": 0,
    "track_last_seen": {},
    "last_summary_frame": 0,  # Track when last summary was printed
    "summary_interval": 30,  # Print summary every N frames
    "loop_video": False,  # Whether to loop video playback
}

# Motion calculation parameters
SMOOTHING_FACTOR = 0.7
MAX_VELOCITY_CHANGE_RATIO = 2.0
MAX_PIXEL_CHANGE = 100
IOU_THRESHOLD = 0.3
MAX_FRAMES_WITHOUT_DETECTION = 10
CONF_THRESHOLD = 0.25


# ============================================================================
# Motion Calculation Functions - Now imported from motion_core.py
# ============================================================================
# All motion functions are imported from motion_core module above
# This maintains backward compatibility with existing code


def resolve_video_path(video_path, search_media=True):
    """Resolve video path - only searches in project media directory."""
    video_path_obj = Path(video_path)
    
    # If absolute path exists, return it
    if video_path_obj.is_absolute() and video_path_obj.exists():
        return video_path_obj
    
    # If relative path exists, return it
    if not video_path_obj.is_absolute() and video_path_obj.exists():
        return video_path_obj.resolve()
    
    if not search_media:
        raise ValueError(f"Video file not found: {video_path}")
    
    # Only search in project media directory
    script_dir = Path(__file__).parent
    project_media = script_dir / "media"
    
    if not project_media.exists():
        raise ValueError(f"Media directory not found: {project_media}")
    
    # Get filename
    video_filename = video_path_obj.name if video_path_obj.name else str(video_path_obj)
    
    # Try exact match first
    video_file = project_media / video_filename
    if video_file.exists() and video_file.is_file():
        return video_file
    
    # Try with .mp4 extension if not present
    if not video_filename.lower().endswith('.mp4'):
        video_file = project_media / f"{video_filename}.mp4"
        if video_file.exists() and video_file.is_file():
            return video_file
    
    # Try case-insensitive match
    for file in project_media.glob("*.mp4"):
        if file.name.lower() == video_filename.lower() or file.name.lower() == f"{video_filename}.mp4".lower():
            return file
    
    raise ValueError(f"Video file not found in {project_media}: {video_path}")


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64


def print_motion_calculations(track_id, frame_count, centroid, velocity, speed, direction, acceleration):
    """Print formatted motion calculations to terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Print header
    print(f"\n{'='*80}")
    print(f"MOTION CALCULATIONS - {timestamp}")
    print(f"{'='*80}")
    print(f"Track ID: {track_id:>3} | Frame: {frame_count:>6}")
    print(f"{'-'*80}")
    
    # Print data in organized columns
    print(f"{'Metric':<20} {'Value':<30} {'Unit':<15}")
    print(f"{'-'*80}")
    print(f"{'Centroid (X, Y)':<20} ({centroid[0]:>7.2f}, {centroid[1]:>7.2f}){'':<10} pixels")
    print(f"{'Velocity (Vx, Vy)':<20} ({velocity[0]:>7.2f}, {velocity[1]:>7.2f}){'':<10} px/s")
    print(f"{'Speed':<20} {speed:>7.2f}{'':<20} px/s")
    print(f"{'Direction':<20} {direction:>7.2f}{'':<20} degrees")
    
    if acceleration:
        accel_mag = math.sqrt(acceleration[0]**2 + acceleration[1]**2)
        print(f"{'Acceleration (Ax, Ay)':<20} ({acceleration[0]:>7.2f}, {acceleration[1]:>7.2f}){'':<10} px/s²")
        print(f"{'Accel Magnitude':<20} {accel_mag:>7.2f}{'':<20} px/s²")
    
    print(f"{'='*80}\n")


def print_tracking_summary(processing_state, frame_count):
    """Print a summary table of all tracked vessels."""
    if not processing_state.get("vessel_tracks"):
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"\n{'='*100}")
    print(f"📋 TRACKING SUMMARY - Frame {frame_count:>6} | {timestamp}")
    print(f"{'='*100}")
    print(f"{'Track ID':<10} {'Centroid (X,Y)':<20} {'Speed':<12} {'Direction':<12} {'Status':<15}")
    print(f"{'-'*100}")
    
    for track_id, track_data in sorted(processing_state["vessel_tracks"].items()):
        if not track_data.get("centroids"):
            continue
        
        centroid = track_data["centroids"][-1]
        speed = track_data["speeds"][-1] if track_data.get("speeds") else 0.0
        direction = track_data["directions"][-1] if track_data.get("directions") else 0.0
        
        # Determine status
        is_selected = track_id in processing_state.get("selected_track_ids", set())
        status = "SELECTED" if is_selected else "TRACKING"
        
        print(f"{track_id:<10} ({centroid[0]:>7.1f}, {centroid[1]:>7.1f}){'':<3} "
              f"{speed:>7.2f} px/s{'':<3} {direction:>7.1f}°{'':<3} {status:<15}")
    
    print(f"{'='*100}\n")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - serve frontend or return API info."""
    # If static files exist, serve index.html
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "NAVIS - Vessel Detection Server", "status": "running", "api": "/docs"}


# ============================================================================
# C3.ai REST Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for C3.ai."""
    wrapper = get_c3_wrapper()
    return wrapper.health_check()


@app.post("/infer")
async def infer(request: dict):
    """
    C3.ai inference endpoint.
    
    Accepts:
    - video_path: str (path to video file)
    - frame_data: str (base64-encoded frame, optional)
    - format: str ("base64" or "raw", optional)
    - frame_limit: int (optional, limit frames to process)
    
    Returns:
    - Detections with metadata
    """
    wrapper = get_c3_wrapper()
    
    if "video_path" in request:
        frame_limit = request.get("frame_limit")
        return wrapper.infer_from_video_path(request["video_path"], frame_limit)
    elif "frame_data" in request:
        format_type = request.get("format", "base64")
        return wrapper.infer_from_frame_data(request["frame_data"], format_type)
    else:
        raise HTTPException(status_code=400, detail="Either 'video_path' or 'frame_data' must be provided")


@app.get("/metrics")
async def metrics():
    """System metrics endpoint for C3.ai."""
    wrapper = get_c3_wrapper()
    return wrapper.get_metrics()


@app.get("/api/videos")
async def list_videos():
    """List available videos in project media directory only."""
    videos = []
    script_dir = Path(__file__).parent
    project_media = script_dir / "media"
    
    if not project_media.exists():
        return {"videos": []}
    
    # Only search in project media directory, only .mp4 files
    for video_file in project_media.glob("*.mp4"):
        if video_file.is_file():
            # Validate video can be opened
            try:
                test_cap = cv2.VideoCapture(str(video_file))
                if test_cap.isOpened():
                    # Try to read first frame to ensure video is valid
                    ret, _ = test_cap.read()
                    test_cap.release()
                    if ret:
                        videos.append({
                            "name": video_file.name,
                            "path": str(video_file),
                            "size": video_file.stat().st_size
                        })
            except Exception as e:
                print(f"Warning: Skipping {video_file.name} - {e}")
                continue
    
    return {"videos": videos}


@app.post("/api/load-video")
async def load_video(data: dict):
    """Load a video file and prepare for processing."""
    global video_cap, processing_state
    
    video_path = data.get("video_path")
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path is required")
    
    try:
        video_path = resolve_video_path(video_path, search_media=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Close existing video if open
    if video_cap is not None:
        video_cap.release()
    
    # Open new video with validation
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Could not open video: {video_path}. Video file may be corrupted or in an unsupported format.")
    
    # Validate video by reading first frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        raise HTTPException(status_code=500, detail=f"Video file appears to be corrupted or empty: {video_path}. Cannot read frames.")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    video_cap = cap
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate video properties
    if width == 0 or height == 0:
        cap.release()
        raise HTTPException(status_code=500, detail=f"Invalid video dimensions: {width}x{height}. Video may be corrupted.")
    
    processing_state.update({
        "video_path": str(video_path),
        "frame_count": 0,
        "fps": fps,
        "width": width,
        "height": height,
        "is_paused": False,  # New video = fresh state, not paused
        "is_tracking": False,
        "auto_playing": True,  # Auto-play a few frames to get detections
        "auto_play_frames": 10,  # Number of frames to auto-play
        "auto_play_count": 0,  # Counter for auto-play
        "selected_track_ids": set(),
        "vessel_tracks": {},
        "active_track_ids": set(),
        "next_track_id": 0,
        "track_last_seen": {},
        "last_summary_frame": 0,  # Reset summary counter
        "summary_interval": 30,  # Print summary every 30 frames
    })
    
    # Read first frame for display
    ret, frame = cap.read()
    if ret and frame is not None:
        processing_state["current_frame"] = frame
        processing_state["frame_count"] = 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    else:
        cap.release()
        raise HTTPException(status_code=500, detail=f"Could not read first frame from video: {video_path}")
    
    return {
        "success": True,
        "video_path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames
    }


@app.post("/api/tag")
async def tag_vessels(data: dict):
    """Tag selected track IDs for tracking."""
    global processing_state
    
    track_ids = data.get("track_ids", [])
    if not track_ids:
        raise HTTPException(status_code=400, detail="track_ids is required")
    
    processing_state["selected_track_ids"] = set(track_ids)
    print(f"Tagged {len(track_ids)} vessel(s): {sorted(track_ids)}")
    
    return {
        "success": True,
        "selected_track_ids": list(processing_state["selected_track_ids"])
    }


@app.post("/api/track")
async def start_tracking():
    """Start tracking selected vessels."""
    global processing_state
    
    selected_ids = processing_state.get("selected_track_ids", set())
    print(f"DEBUG: Checking selected_track_ids: {selected_ids}, type: {type(selected_ids)}, empty: {not selected_ids}")
    
    if not selected_ids or len(selected_ids) == 0:
        raise HTTPException(status_code=400, detail="No vessels selected. Please tag vessels first.")
    
    processing_state["is_tracking"] = True
    processing_state["is_paused"] = False
    print(f"Started tracking {len(selected_ids)} vessel(s): {sorted(selected_ids)}")
    
    return {
        "success": True,
        "message": "Tracking started",
        "selected_track_ids": list(selected_ids)
    }


@app.post("/api/pause")
async def pause_tracking():
    """Pause video tracking."""
    global processing_state
    processing_state["is_paused"] = True
    processing_state["is_tracking"] = False
    
    return {
        "success": True,
        "message": "Tracking paused"
    }


@app.post("/api/loop")
async def toggle_loop(data: dict):
    """Toggle video loop mode."""
    global processing_state
    
    loop_enabled = data.get("loop", False)
    processing_state["loop_video"] = loop_enabled
    
    return {
        "success": True,
        "loop_enabled": loop_enabled,
        "message": "Video loop " + ("enabled" if loop_enabled else "disabled")
    }


@app.post("/api/all-detections")
async def start_all_detections():
    """Start tracking all detected vessels automatically."""
    global processing_state
    
    # Set mode to track all vessels
    processing_state["is_tracking"] = True
    processing_state["is_paused"] = False
    # Don't filter by selected_track_ids - track all active vessels
    print("Started tracking all detections")
    
    return {
        "success": True,
        "message": "Tracking all detections"
    }


# ============================================================================
# WebSocket Connection
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming frames and detection data."""
    global processing_state, yolo_model
    
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # Load YOLO model if not loaded
        if yolo_model is None:
            model_path = Path("models/yolov8n_vessels.pt")
            if not model_path.exists():
                await websocket.send_json({
                    "type": "error",
                    "message": f"YOLO model not found: {model_path}"
                })
                return
            yolo_model = YOLO(str(model_path))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            yolo_model.to(device)
            await websocket.send_json({
                "type": "model_loaded",
                "device": device
            })
        
        # Send first frame if available
        if processing_state["current_frame"] is not None and processing_state["frame_count"] == 1:
            frame = processing_state["current_frame"]
            frame_base64 = frame_to_base64(frame)
            
            # Run YOLO on first frame
            detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
            
            # Assign track IDs to detections
            detections_with_ids = []
            for i, det in enumerate(detections):
                track_id = processing_state["next_track_id"]
                processing_state["next_track_id"] += 1
                bbox = det[:4]
                centroid = compute_bbox_centroid(bbox)
                detections_with_ids.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": det[4],
                    "speed": None,  # No speed yet on first frame
                    "centroid": centroid,
                    "centroid_history": [centroid]  # Just current centroid
                })
                # Initialize track
                bbox = det[:4]
                centroid = compute_bbox_centroid(bbox)
                processing_state["vessel_tracks"][track_id] = {
                    "centroids": [centroid],
                    "velocities": [],
                    "speeds": [],
                    "accelerations": [],
                    "directions": [],
                    "bboxes": [bbox],
                    "frames": [processing_state["frame_count"]]
                }
                processing_state["active_track_ids"].add(track_id)
                processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
            
            await websocket.send_json({
                "type": "frame",
                "frame": frame_base64,
                "frame_count": processing_state["frame_count"],
                "detections": detections_with_ids,
                "width": processing_state["width"],
                "height": processing_state["height"]
            })
        
        # Main processing loop
        while True:
            if processing_state["video_path"] is None or video_cap is None:
                await asyncio.sleep(0.1)
                continue
            
            # Handle auto-play state - play a few frames to gather detections
            if processing_state.get("auto_playing", False) and not processing_state["is_tracking"]:
                # Read next frame
                ret, frame = video_cap.read()
                if not ret:
                    # End of video during auto-play - stop and show last frame
                    processing_state["auto_playing"] = False
                    processing_state["is_paused"] = True
                    continue
                
                processing_state["frame_count"] += 1
                processing_state["current_frame"] = frame.copy()
                processing_state["auto_play_count"] += 1
                dt = 1.0 / processing_state["fps"]
                
                # Run YOLO detection
                detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
                
                # Match detections to existing tracks or create new ones
                if processing_state["vessel_tracks"]:
                    matches, unmatched = match_detections_to_tracks(
                        detections, 
                        processing_state["vessel_tracks"], 
                        IOU_THRESHOLD
                    )
                    
                    # Update matched tracks
                    for track_id, det_idx in matches.items():
                        bbox = detections[det_idx][:4]
                        centroid = compute_bbox_centroid(bbox)
                        track_data = processing_state["vessel_tracks"][track_id]
                        track_data["bboxes"].append(bbox)
                        track_data["centroids"].append(centroid)
                        track_data["frames"].append(processing_state["frame_count"])
                        processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                    
                    # Create new tracks for unmatched detections
                    for det_idx in unmatched:
                        track_id = processing_state["next_track_id"]
                        processing_state["next_track_id"] += 1
                        bbox = detections[det_idx][:4]
                        centroid = compute_bbox_centroid(bbox)
                        processing_state["vessel_tracks"][track_id] = {
                            "centroids": [centroid],
                            "velocities": [],
                            "speeds": [],
                            "accelerations": [],
                            "directions": [],
                            "bboxes": [bbox],
                            "frames": [processing_state["frame_count"]]
                        }
                        processing_state["active_track_ids"].add(track_id)
                        processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                else:
                    # No existing tracks - assign new IDs
                    for det in detections:
                        track_id = processing_state["next_track_id"]
                        processing_state["next_track_id"] += 1
                        bbox = det[:4]
                        centroid = compute_bbox_centroid(bbox)
                        processing_state["vessel_tracks"][track_id] = {
                            "centroids": [centroid],
                            "velocities": [],
                            "speeds": [],
                            "accelerations": [],
                            "directions": [],
                            "bboxes": [bbox],
                            "frames": [processing_state["frame_count"]]
                        }
                        processing_state["active_track_ids"].add(track_id)
                        processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                
                # Build detections list for display
                detections_with_ids = []
                for track_id in processing_state["active_track_ids"]:
                    if track_id in processing_state["vessel_tracks"]:
                        track_data = processing_state["vessel_tracks"][track_id]
                        if track_data["bboxes"]:
                            last_bbox = track_data["bboxes"][-1]
                            centroid = track_data["centroids"][-1] if track_data["centroids"] else None
                            speed = track_data["speeds"][-1] if track_data["speeds"] else None
                            centroid_history = track_data["centroids"][-20:] if len(track_data["centroids"]) > 1 else (track_data["centroids"] if track_data["centroids"] else [])
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": last_bbox,
                                "confidence": 0.9,  # Approximate confidence
                                "speed": speed,
                                "centroid": centroid,
                                "centroid_history": centroid_history
                            })
                
                # Send frame
                frame_base64 = frame_to_base64(frame)
                await websocket.send_json({
                    "type": "frame",
                    "frame": frame_base64,
                    "frame_count": processing_state["frame_count"],
                    "detections": detections_with_ids,
                    "width": processing_state["width"],
                    "height": processing_state["height"]
                })
                
                # Check if we've played enough frames
                if processing_state["auto_play_count"] >= processing_state["auto_play_frames"]:
                    processing_state["auto_playing"] = False
                    processing_state["is_paused"] = True
                    print(f"Auto-played {processing_state['auto_play_count']} frames. Ready for tagging.")
                    await websocket.send_json({
                        "type": "auto_play_complete",
                        "message": f"Played {processing_state['auto_play_count']} frames. Ready to tag vessels.",
                        "frame_count": processing_state["frame_count"]
                    })
                
                # Control frame rate during auto-play
                await asyncio.sleep(1.0 / processing_state["fps"])
                continue
            
            # Handle ready state (not paused, not tracking) - show current frame with all detections
            if not processing_state["is_paused"] and not processing_state["is_tracking"]:
                if processing_state["current_frame"] is not None:
                    frame = processing_state["current_frame"]
                    # Run YOLO to get detections
                    detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
                    
                    # Match to existing tracks or create new ones
                    detections_with_ids = []
                    if processing_state["vessel_tracks"]:
                        matches, unmatched = match_detections_to_tracks(
                            detections, 
                            processing_state["vessel_tracks"], 
                            IOU_THRESHOLD
                        )
                        
                        # Add matched detections
                        for track_id, det_idx in matches.items():
                            track_data = processing_state["vessel_tracks"].get(track_id, {})
                            bbox = detections[det_idx][:4]
                            centroid = track_data.get("centroids", [])
                            centroid = centroid[-1] if centroid else compute_bbox_centroid(bbox)
                            speed = track_data.get("speeds", [])
                            speed = speed[-1] if speed else None
                            centroid_history = track_data.get("centroids", [])[-20:] if len(track_data.get("centroids", [])) > 1 else (track_data.get("centroids", []) if track_data.get("centroids") else [])
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": speed,
                                "centroid": centroid,
                                "centroid_history": centroid_history
                            })
                        
                        # Create new tracks for unmatched detections
                        for det_idx in unmatched:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = detections[det_idx][:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    else:
                        # No existing tracks - assign new IDs
                        for det in detections:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = det[:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": det[4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    
                    frame_base64 = frame_to_base64(frame)
                    await websocket.send_json({
                        "type": "frame",
                        "frame": frame_base64,
                        "frame_count": processing_state["frame_count"],
                        "detections": detections_with_ids,
                        "width": processing_state["width"],
                        "height": processing_state["height"]
                    })
                
                await asyncio.sleep(0.5)  # Update every 0.5s when ready
                continue
            
            # Handle pause state - send current frame with all detections for re-tagging
            if processing_state["is_paused"]:
                if processing_state["current_frame"] is not None:
                    frame = processing_state["current_frame"]
                    # Run YOLO to get fresh detections
                    detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
                    
                    # Match to existing tracks to get track IDs
                    detections_with_ids = []
                    if processing_state["vessel_tracks"]:
                        matches, unmatched = match_detections_to_tracks(
                            detections, 
                            processing_state["vessel_tracks"], 
                            IOU_THRESHOLD
                        )
                        
                        # Add matched detections
                        for track_id, det_idx in matches.items():
                            track_data = processing_state["vessel_tracks"].get(track_id, {})
                            bbox = detections[det_idx][:4]
                            centroid = track_data.get("centroids", [])
                            centroid = centroid[-1] if centroid else compute_bbox_centroid(bbox)
                            speed = track_data.get("speeds", [])
                            speed = speed[-1] if speed else None
                            centroid_history = track_data.get("centroids", [])[-20:] if len(track_data.get("centroids", [])) > 1 else (track_data.get("centroids", []) if track_data.get("centroids") else [])
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": speed,
                                "centroid": centroid,
                                "centroid_history": centroid_history
                            })
                        
                        # Create new tracks for unmatched detections
                        for det_idx in unmatched:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = detections[det_idx][:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": detections[det_idx][4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    else:
                        # No existing tracks - assign new IDs
                        for det in detections:
                            track_id = processing_state["next_track_id"]
                            processing_state["next_track_id"] += 1
                            bbox = det[:4]
                            centroid = compute_bbox_centroid(bbox)
                            processing_state["vessel_tracks"][track_id] = {
                                "centroids": [centroid],
                                "velocities": [],
                                "speeds": [],
                                "accelerations": [],
                                "directions": [],
                                "bboxes": [bbox],
                                "frames": [processing_state["frame_count"]]
                            }
                            processing_state["active_track_ids"].add(track_id)
                            processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                            detections_with_ids.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "confidence": det[4],
                                "speed": None,  # New track, no speed yet
                                "centroid": centroid,
                                "centroid_history": [centroid]
                            })
                    
                    frame_base64 = frame_to_base64(frame)
                    await websocket.send_json({
                        "type": "frame",
                        "frame": frame_base64,
                        "frame_count": processing_state["frame_count"],
                        "detections": detections_with_ids,
                        "width": processing_state["width"],
                        "height": processing_state["height"]
                    })
                
                await asyncio.sleep(0.5)  # Update every 0.5s when paused
                continue
            
            # Handle tracking state
            if not processing_state["is_tracking"]:
                await asyncio.sleep(0.1)
                continue
            
            # Read next frame
            ret, frame = video_cap.read()
            if not ret:
                # End of video - check if looping
                if processing_state.get("loop_video", False):
                    # Reset to beginning and continue
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    processing_state["frame_count"] = 0
                    processing_state["vessel_tracks"] = {}  # Reset tracks for new loop
                    processing_state["active_track_ids"] = set()
                    processing_state["next_track_id"] = 0
                    processing_state["track_last_seen"] = {}
                    print(f"\nVideo looped - restarting from frame 0")
                    ret, frame = video_cap.read()
                    if not ret:
                        # Still can't read, stop
                        processing_state["is_tracking"] = False
                        processing_state["is_paused"] = True
                        continue
                else:
                    # End of video - stop tracking
                    processing_state["is_tracking"] = False
                    processing_state["is_paused"] = True
                    print(f"\nVideo ended at frame {processing_state['frame_count']}. Tracking stopped.")
                    await websocket.send_json({
                        "type": "video_ended",
                        "message": "Video playback completed",
                        "frame_count": processing_state["frame_count"]
                    })
                    # Stay in pause state
                    continue
            
            processing_state["frame_count"] += 1
            processing_state["current_frame"] = frame.copy()
            dt = 1.0 / processing_state["fps"]
            
            # Run YOLO detection
            detections = detect_vessels_with_yolo(frame, yolo_model, CONF_THRESHOLD)
            
            # Check if we're in "all detections" mode (no selected_track_ids filter)
            # or "target vessels" mode (only track selected ones)
            if processing_state["selected_track_ids"]:
                # Target vessels mode - only track selected ones
                selected_tracks = {
                    tid: processing_state["vessel_tracks"][tid]
                    for tid in processing_state["selected_track_ids"]
                    if tid in processing_state["vessel_tracks"]
                }
                
                if selected_tracks and detections:
                    matches, unmatched = match_detections_to_tracks(detections, selected_tracks, IOU_THRESHOLD)
                else:
                    matches = {}
                    unmatched = list(range(len(detections)))
            else:
                # All detections mode - track all vessels
                all_tracks = processing_state["vessel_tracks"]
                if all_tracks and detections:
                    matches, unmatched = match_detections_to_tracks(detections, all_tracks, IOU_THRESHOLD)
                else:
                    matches = {}
                    unmatched = list(range(len(detections)))
            
            # Update matched tracks
            detections_with_ids = []
            for track_id, det_idx in matches.items():
                # In target vessels mode, only process selected tracks
                # In all detections mode, process all tracks
                if processing_state["selected_track_ids"] and track_id not in processing_state["selected_track_ids"]:
                    continue
                
                bbox = detections[det_idx][:4]
                centroid = compute_bbox_centroid(bbox)
                
                track_data = processing_state["vessel_tracks"][track_id]
                track_data["bboxes"].append(bbox)
                track_data["centroids"].append(centroid)
                track_data["frames"].append(processing_state["frame_count"])
                processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                
                # Compute motion (if we have previous centroid)
                if len(track_data["centroids"]) > 1:
                    prev_centroid = track_data["centroids"][-2]
                    curr_centroid = track_data["centroids"][-1]
                    
                    if validate_centroid_change(prev_centroid, curr_centroid, MAX_PIXEL_CHANGE):
                        raw_velocity = compute_velocity(prev_centroid, curr_centroid, dt)
                        velocity = smooth_velocity(raw_velocity, track_data["velocities"], 
                                                  SMOOTHING_FACTOR, MAX_VELOCITY_CHANGE_RATIO)
                        speed = compute_speed(velocity)
                        direction = compute_direction(velocity)
                        
                        track_data["velocities"].append(velocity)
                        track_data["speeds"].append(speed)
                        track_data["directions"].append(direction)
                        
                        # Compute acceleration
                        if len(track_data["velocities"]) > 1:
                            prev_velocity = track_data["velocities"][-2]
                            curr_velocity = track_data["velocities"][-1]
                            acceleration = compute_acceleration(prev_velocity, curr_velocity, dt)
                            track_data["accelerations"].append(acceleration)
                            
                            # Print motion calculations to terminal with formatted output
                            print_motion_calculations(
                                track_id=track_id,
                                frame_count=processing_state["frame_count"],
                                centroid=curr_centroid,
                                velocity=velocity,
                                speed=speed,
                                direction=direction,
                                acceleration=acceleration
                            )
                            
                            # Send motion calculations to frontend terminal
                            accel_mag = math.sqrt(acceleration[0]**2 + acceleration[1]**2) if acceleration else 0.0
                            await websocket.send_json({
                                "type": "motion_calculation",
                                "track_id": track_id,
                                "frame_count": processing_state["frame_count"],
                                "centroid": curr_centroid,
                                "velocity": velocity,
                                "speed": speed,
                                "direction": direction,
                                "acceleration": acceleration,
                                "acceleration_magnitude": accel_mag,
                                "timestamp": datetime.now().isoformat()
                            })
                
                # Get speed and centroid history for this track
                speed = track_data["speeds"][-1] if track_data["speeds"] else None
                centroid_history = track_data["centroids"][-20:] if len(track_data["centroids"]) > 1 else []  # Last 20 centroids
                
                detections_with_ids.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": detections[det_idx][4],
                    "speed": speed,
                    "centroid": centroid,
                    "centroid_history": centroid_history
                })
            
            # Create new tracks for unmatched detections (only in all detections mode)
            if not processing_state["selected_track_ids"]:  # All detections mode
                for det_idx in unmatched:
                    detection = detections[det_idx]
                    track_id = processing_state["next_track_id"]
                    processing_state["next_track_id"] += 1
                    bbox = detection[:4]
                    centroid = compute_bbox_centroid(bbox)
                    processing_state["vessel_tracks"][track_id] = {
                        "centroids": [centroid],
                        "velocities": [],
                        "speeds": [],
                        "accelerations": [],
                        "directions": [],
                        "bboxes": [bbox],
                        "frames": [processing_state["frame_count"]]
                    }
                    processing_state["active_track_ids"].add(track_id)
                    processing_state["track_last_seen"][track_id] = processing_state["frame_count"]
                    detections_with_ids.append({
                        "track_id": track_id,
                        "bbox": bbox,
                        "confidence": detection[4],
                        "speed": None,  # New track, no speed yet
                        "centroid": centroid,
                        "centroid_history": [centroid]
                    })
            
            # Remove tracks that haven't been seen
            tracks_to_remove = []
            for track_id in list(processing_state["active_track_ids"]):
                # In target vessels mode, only manage selected tracks
                # In all detections mode, manage all tracks
                if processing_state["selected_track_ids"] and track_id not in processing_state["selected_track_ids"]:
                    continue
                if track_id in processing_state["track_last_seen"]:
                    frames_since_seen = processing_state["frame_count"] - processing_state["track_last_seen"][track_id]
                    if frames_since_seen > MAX_FRAMES_WITHOUT_DETECTION:
                        tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                processing_state["active_track_ids"].discard(track_id)
                if track_id in processing_state["vessel_tracks"]:
                    del processing_state["vessel_tracks"][track_id]
                if track_id in processing_state["track_last_seen"]:
                    del processing_state["track_last_seen"][track_id]
            
            # Filter detections based on mode
            if processing_state["selected_track_ids"]:
                # Target vessels mode - only show selected ones
                filtered_detections = [
                    det for det in detections_with_ids 
                    if det["track_id"] in processing_state["selected_track_ids"]
                ]
            else:
                # All detections mode - show all
                filtered_detections = detections_with_ids
            
            # Print periodic summary of all tracked vessels
            if processing_state["frame_count"] - processing_state["last_summary_frame"] >= processing_state["summary_interval"]:
                print_tracking_summary(processing_state, processing_state["frame_count"])
                processing_state["last_summary_frame"] = processing_state["frame_count"]
            
            # Send frame and detections (only selected ones)
            frame_base64 = frame_to_base64(frame)
            await websocket.send_json({
                "type": "frame",
                "frame": frame_base64,
                "frame_count": processing_state["frame_count"],
                "detections": filtered_detections,
                "width": processing_state["width"],
                "height": processing_state["height"]
            })
            
            # Control frame rate
            await asyncio.sleep(1.0 / processing_state["fps"])
    
    except WebSocketDisconnect:
        active_connections.discard(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        active_connections.discard(websocket)


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat interface with agent."""
    global orchestrator, processing_state
    
    await websocket.accept()
    chat_connections.add(websocket)
    
    # Initialize orchestrator if not already initialized
    if orchestrator is None:
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
    
    try:
        await websocket.send_json({
            "type": "system",
            "message": "Welcome to NAVIS (Nautical Analysis & Vessel Intelligence System). I'm ready to help you analyze vessel tracking data. Ask me anything about tracked vessels, motion metrics, or request visualizations!"
        })
        
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", data)
            except json.JSONDecodeError:
                user_message = data
            
            # Sync tracks from processing_state before handling chat
            if processing_state.get("vessel_tracks"):
                orchestrator.sync_tracks_from_processing_state(
                    processing_state["vessel_tracks"],
                    processing_state.get("frame_count", 0),
                    processing_state.get("fps", 30.0),
                    processing_state.get("video_path")
                )
            
            # Create chat request
            chat_request = ChatRequest(message=user_message)
            
            # Handle chat with orchestrator
            chat_response = orchestrator.handle_chat(chat_request)
            
            # Build response JSON
            response_json = {
                "type": "agent",
                "message": chat_response.response,
                "success": chat_response.success,
                "error": chat_response.error,
                "metadata": chat_response.metadata
            }
            
            # Include plot_data if present
            if chat_response.plot_data:
                response_json["plot_data"] = chat_response.plot_data
            
            await websocket.send_json(response_json)
    
    except WebSocketDisconnect:
        chat_connections.discard(websocket)
    except Exception as e:
        print(f"Chat WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error: {str(e)}"
            })
        except:
            pass
        chat_connections.discard(websocket)


# Initialize orchestrator on startup
@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on server startup."""
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
        print("Orchestrator initialized")


# Serve SPA for all non-API routes (must be last route)
# This catch-all route must be defined after all other routes
if static_path.exists():
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA for all non-API routes."""
        # Don't serve SPA for API routes, docs, or WebSocket
        if (full_path.startswith("api") or 
            full_path.startswith("docs") or 
            full_path.startswith("openapi.json") or 
            full_path.startswith("ws") or
            full_path == "health" or
            full_path == "infer" or
            full_path == "metrics"):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Serve index.html for all other routes (SPA routing)
        index_path = static_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(status_code=404, detail="Frontend not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

