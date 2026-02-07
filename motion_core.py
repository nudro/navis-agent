"""
Motion Core Library - Stable motion calculation functions.
These functions are used by both the GUI and the agent system.
Keep function signatures intact for backward compatibility.
"""

import cv2
import numpy as np
import math
import torch
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO


def compute_bbox_centroid(bbox):
    """Compute centroid from bounding box [x1, y1, x2, y2]."""
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    centroid_x = (x1 + x2) / 2.0
    centroid_y = (y1 + y2) / 2.0
    return (centroid_x, centroid_y)


def compute_bbox_area(bbox):
    """Compute area of bounding box in pixels."""
    if bbox is None or len(bbox) < 4:
        return 0
    x1, y1, x2, y2 = bbox[:4]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return int(width * height)


def detect_vessels_with_yolo(frame, yolo_model, conf_threshold=0.25):
    """Use YOLO to detect vessels in a frame and return bounding boxes."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = yolo_model(frame, conf=conf_threshold, verbose=False, device=device)
    bboxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            bboxes.append([int(x1), int(y1), int(x2), int(y2), confidence])
    
    return bboxes


def compute_velocity(prev_centroid, curr_centroid, dt):
    """Compute velocity vector from centroid positions."""
    if prev_centroid is None or curr_centroid is None:
        return None
    vx = (curr_centroid[0] - prev_centroid[0]) / dt
    vy = (curr_centroid[1] - prev_centroid[1]) / dt
    return (vx, vy)


def compute_speed(velocity):
    """Compute speed magnitude from velocity vector."""
    if velocity is None:
        return None
    return math.sqrt(velocity[0]**2 + velocity[1]**2)


def compute_acceleration(prev_velocity, curr_velocity, dt):
    """Compute acceleration vector from velocity vectors."""
    if prev_velocity is None or curr_velocity is None:
        return None
    ax = (curr_velocity[0] - prev_velocity[0]) / dt
    # Bug fix: was using curr_velocity[1] twice, now correctly uses prev_velocity[1]
    ay = (curr_velocity[1] - prev_velocity[1]) / dt
    return (ax, ay)


def compute_direction(velocity):
    """Compute direction angle in degrees from velocity vector."""
    if velocity is None:
        return None
    angle_rad = math.atan2(velocity[1], velocity[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def smooth_velocity(curr_velocity, prev_velocities, smoothing_factor=0.7, max_change_ratio=2.0):
    """Smooth velocity using exponential moving average and outlier detection."""
    if curr_velocity is None:
        return None
    if not prev_velocities:
        return curr_velocity
    
    prev_velocity = prev_velocities[-1]
    curr_speed = math.sqrt(curr_velocity[0]**2 + curr_velocity[1]**2)
    prev_speed = math.sqrt(prev_velocity[0]**2 + prev_velocity[1]**2)
    
    if prev_speed > 0:
        speed_change_ratio = curr_speed / prev_speed
        if speed_change_ratio > max_change_ratio or speed_change_ratio < (1.0 / max_change_ratio):
            smoothed_vx = prev_velocity[0] * (1 - smoothing_factor) + curr_velocity[0] * smoothing_factor
            smoothed_vy = prev_velocity[1] * (1 - smoothing_factor) + curr_velocity[1] * smoothing_factor
            return (smoothed_vx, smoothed_vy)
    
    smoothed_vx = prev_velocity[0] * (1 - smoothing_factor) + curr_velocity[0] * smoothing_factor
    smoothed_vy = prev_velocity[1] * (1 - smoothing_factor) + curr_velocity[1] * smoothing_factor
    return (smoothed_vx, smoothed_vy)


def validate_centroid_change(prev_centroid, curr_centroid, max_pixel_change=100):
    """Validate that centroid change is reasonable (outlier detection)."""
    if prev_centroid is None or curr_centroid is None:
        return True
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    distance = math.sqrt(dx**2 + dy**2)
    return distance <= max_pixel_change


def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections_to_tracks(new_detections, existing_tracks, iou_threshold=0.3):
    """Match new YOLO detections to existing tracks using IoU."""
    matches = {}  # {track_id: detection_index}
    unmatched_detections = list(range(len(new_detections)))
    
    if not existing_tracks or not new_detections:
        return matches, unmatched_detections
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(existing_tracks), len(new_detections)))
    
    for track_idx, (track_id, track_data) in enumerate(existing_tracks.items()):
        if not track_data.get("bboxes"):
            continue
        prev_bbox = track_data["bboxes"][-1]
        
        for det_idx, det_bbox in enumerate(new_detections):
            iou = compute_iou(prev_bbox[:4], det_bbox[:4])
            iou_matrix[track_idx, det_idx] = iou
    
    # Greedy matching
    used_tracks = set()
    used_detections = set()
    matches_list = []
    
    for track_idx in range(len(existing_tracks)):
        for det_idx in range(len(new_detections)):
            if iou_matrix[track_idx, det_idx] >= iou_threshold:
                matches_list.append((track_idx, det_idx, iou_matrix[track_idx, det_idx]))
    
    matches_list.sort(key=lambda x: x[2], reverse=True)
    
    for track_idx, det_idx, iou in matches_list:
        if track_idx not in used_tracks and det_idx not in used_detections:
            track_id = list(existing_tracks.keys())[track_idx]
            matches[track_id] = det_idx
            used_tracks.add(track_idx)
            used_detections.add(det_idx)
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)
    
    return matches, unmatched_detections


def compute_centroid_distance_px(c1, c2):
    """Euclidean distance between two centroids in pixels.
    
    Args:
        c1: First centroid as (x, y) tuple
        c2: Second centroid as (x, y) tuple
        
    Returns:
        Distance in pixels, or None if either centroid is None
    """
    if c1 is None or c2 is None:
        return None
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    return math.sqrt(dx*dx + dy*dy)


def compute_bbox_edge_distance_px(b1, b2):
    """Minimum distance between two axis-aligned bboxes in pixels.
    Returns 0 if they overlap.
    
    Args:
        b1: First bounding box as [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
        b2: Second bounding box as [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
        
    Returns:
        Minimum edge distance in pixels, or None if either bbox is None
    """
    if b1 is None or b2 is None:
        return None
    x1a, y1a, x2a, y2a = b1[:4]
    x1b, y1b, x2b, y2b = b2[:4]

    # separation along x
    if x2a < x1b:
        dx = x1b - x2a
    elif x2b < x1a:
        dx = x1a - x2b
    else:
        dx = 0

    # separation along y
    if y2a < y1b:
        dy = y1b - y2a
    elif y2b < y1a:
        dy = y1a - y2b
    else:
        dy = 0

    return math.sqrt(dx*dx + dy*dy)
