"""
Unit tests for the two-agent system.
Tests motion core functions, CV Agent, Orchestrator, and schemas.
"""

import unittest
import math
import numpy as np
from typing import Dict, List, Any

from motion_core import (
    compute_bbox_centroid,
    compute_bbox_area,
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
from schemas import (
    CVAgentRequest,
    CVAgentResponse,
    FrameState,
    TrackState,
    TrackDiagnostics,
    CVAgentConfig,
    EventSummary,
    OrchestratorMetrics,
)
from cv_agent import CVAgent
from orchestrator import Orchestrator


class TestMotionCore(unittest.TestCase):
    """Test motion core functions."""
    
    def test_compute_bbox_centroid(self):
        """Test centroid calculation."""
        bbox = [10, 20, 50, 60]
        centroid = compute_bbox_centroid(bbox)
        self.assertEqual(centroid, (30.0, 40.0))
        
        # Test None/empty
        self.assertIsNone(compute_bbox_centroid(None))
        self.assertIsNone(compute_bbox_centroid([10, 20]))
    
    def test_compute_bbox_area(self):
        """Test bbox area calculation."""
        bbox = [10, 20, 50, 60]
        area = compute_bbox_area(bbox)
        self.assertEqual(area, 1600)  # 40 * 40
        
        # Test invalid
        self.assertEqual(compute_bbox_area(None), 0)
    
    def test_compute_velocity(self):
        """Test velocity calculation."""
        prev_centroid = (10.0, 20.0)
        curr_centroid = (20.0, 30.0)
        dt = 1.0
        
        velocity = compute_velocity(prev_centroid, curr_centroid, dt)
        self.assertEqual(velocity, (10.0, 10.0))
        
        # Test None
        self.assertIsNone(compute_velocity(None, curr_centroid, dt))
    
    def test_compute_speed(self):
        """Test speed calculation."""
        velocity = (3.0, 4.0)
        speed = compute_speed(velocity)
        self.assertEqual(speed, 5.0)  # 3-4-5 triangle
        
        # Test None
        self.assertIsNone(compute_speed(None))
    
    def test_compute_acceleration(self):
        """Test acceleration calculation (with bug fix)."""
        prev_velocity = (10.0, 20.0)
        curr_velocity = (15.0, 25.0)
        dt = 1.0
        
        acceleration = compute_acceleration(prev_velocity, curr_velocity, dt)
        # Should be (5.0, 5.0) - bug was using curr_velocity[1] twice
        self.assertEqual(acceleration, (5.0, 5.0))
        
        # Test None
        self.assertIsNone(compute_acceleration(None, curr_velocity, dt))
    
    def test_compute_direction(self):
        """Test direction calculation."""
        velocity = (1.0, 0.0)  # Moving right
        direction = compute_direction(velocity)
        self.assertAlmostEqual(direction, 0.0, places=1)
        
        velocity = (0.0, 1.0)  # Moving down
        direction = compute_direction(velocity)
        self.assertAlmostEqual(direction, 90.0, places=1)
        
        # Test None
        self.assertIsNone(compute_direction(None))
    
    def test_smooth_velocity(self):
        """Test velocity smoothing."""
        curr_velocity = (10.0, 10.0)
        prev_velocities = [(5.0, 5.0)]
        smoothing_factor = 0.7
        
        smoothed = smooth_velocity(curr_velocity, prev_velocities, smoothing_factor)
        # Should be weighted average
        expected_vx = 5.0 * 0.3 + 10.0 * 0.7
        expected_vy = 5.0 * 0.3 + 10.0 * 0.7
        self.assertAlmostEqual(smoothed[0], expected_vx, places=2)
        self.assertAlmostEqual(smoothed[1], expected_vy, places=2)
        
        # Test None
        self.assertIsNone(smooth_velocity(None, prev_velocities, smoothing_factor))
        
        # Test empty prev_velocities
        self.assertEqual(smooth_velocity(curr_velocity, [], smoothing_factor), curr_velocity)
    
    def test_validate_centroid_change(self):
        """Test centroid change validation."""
        prev_centroid = (10.0, 20.0)
        curr_centroid = (15.0, 25.0)  # Distance = sqrt(50) ≈ 7.07
        max_pixel_change = 100
        
        self.assertTrue(validate_centroid_change(prev_centroid, curr_centroid, max_pixel_change))
        
        # Test large change
        curr_centroid = (200.0, 200.0)  # Distance > 100
        self.assertFalse(validate_centroid_change(prev_centroid, curr_centroid, max_pixel_change))
        
        # Test None
        self.assertTrue(validate_centroid_change(None, curr_centroid, max_pixel_change))
    
    def test_compute_iou(self):
        """Test IoU calculation."""
        bbox1 = [10, 10, 50, 50]  # 40x40 = 1600
        bbox2 = [30, 30, 70, 70]  # 40x40 = 1600
        
        # Intersection: [30, 30, 50, 50] = 20x20 = 400
        # Union: 1600 + 1600 - 400 = 2800
        # IoU = 400 / 2800 ≈ 0.143
        iou = compute_iou(bbox1, bbox2)
        self.assertAlmostEqual(iou, 400.0 / 2800.0, places=3)
        
        # Test no overlap
        bbox3 = [100, 100, 150, 150]
        iou = compute_iou(bbox1, bbox3)
        self.assertEqual(iou, 0.0)
        
        # Test identical
        iou = compute_iou(bbox1, bbox1)
        self.assertEqual(iou, 1.0)
    
    def test_compute_centroid_distance_px(self):
        """Test centroid distance calculation."""
        c1 = (10.0, 20.0)
        c2 = (30.0, 40.0)
        
        # Distance = sqrt((30-10)^2 + (40-20)^2) = sqrt(400 + 400) = sqrt(800) ≈ 28.28
        distance = compute_centroid_distance_px(c1, c2)
        self.assertAlmostEqual(distance, math.sqrt(800), places=2)
        
        # Test None
        self.assertIsNone(compute_centroid_distance_px(None, c2))
        self.assertIsNone(compute_centroid_distance_px(c1, None))
        
        # Test same point
        distance = compute_centroid_distance_px(c1, c1)
        self.assertEqual(distance, 0.0)
    
    def test_compute_bbox_edge_distance_px(self):
        """Test bbox edge distance calculation."""
        # Overlapping bboxes (should return 0)
        bbox1 = [10, 10, 50, 50]
        bbox2 = [30, 30, 70, 70]
        distance = compute_bbox_edge_distance_px(bbox1, bbox2)
        self.assertEqual(distance, 0.0)
        
        # Non-overlapping bboxes
        bbox3 = [100, 100, 150, 150]  # Far from bbox1
        # bbox1 ends at (50, 50), bbox3 starts at (100, 100)
        # Distance = sqrt((100-50)^2 + (100-50)^2) = sqrt(2500 + 2500) = sqrt(5000) ≈ 70.71
        distance = compute_bbox_edge_distance_px(bbox1, bbox3)
        self.assertAlmostEqual(distance, math.sqrt(5000), places=2)
        
        # Horizontally separated
        bbox4 = [60, 10, 100, 50]  # To the right of bbox1
        # bbox1 ends at x=50, bbox4 starts at x=60, so dx=10, dy=0
        distance = compute_bbox_edge_distance_px(bbox1, bbox4)
        self.assertEqual(distance, 10.0)
        
        # Vertically separated
        bbox5 = [10, 60, 50, 100]  # Below bbox1
        # bbox1 ends at y=50, bbox5 starts at y=60, so dx=0, dy=10
        distance = compute_bbox_edge_distance_px(bbox1, bbox5)
        self.assertEqual(distance, 10.0)
        
        # Test None
        self.assertIsNone(compute_bbox_edge_distance_px(None, bbox2))
        self.assertIsNone(compute_bbox_edge_distance_px(bbox1, None))
        
        # Test with confidence in bbox (should still work)
        bbox6 = [60, 10, 100, 50, 0.9]
        distance = compute_bbox_edge_distance_px(bbox1, bbox6)
        self.assertEqual(distance, 10.0)


class TestAssociation(unittest.TestCase):
    """Test detection-to-track association."""
    
    def test_match_detections_to_tracks(self):
        """Test association correctness."""
        # Existing track at position [10, 10, 50, 50]
        existing_tracks = {
            0: {
                "bboxes": [[10, 10, 50, 50]],
                "centroids": [(30, 30)],
            }
        }
        
        # Detection close to existing track (should match)
        new_detections = [
            [12, 12, 52, 52, 0.9]  # High IoU with track 0
        ]
        
        matches, unmatched = match_detections_to_tracks(
            new_detections,
            existing_tracks,
            iou_threshold=0.3
        )
        
        self.assertIn(0, matches)
        self.assertEqual(len(unmatched), 0)
        
        # Detection far from existing track (should not match)
        new_detections = [
            [200, 200, 250, 250, 0.9]  # Low IoU with track 0
        ]
        
        matches, unmatched = match_detections_to_tracks(
            new_detections,
            existing_tracks,
            iou_threshold=0.3
        )
        
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unmatched), 1)
        self.assertIn(0, unmatched)


class TestSchemas(unittest.TestCase):
    """Test Pydantic schemas."""
    
    def test_frame_state(self):
        """Test FrameState schema."""
        frame_state = FrameState(
            frame_number=1,
            bbox=[10, 20, 50, 60],
            centroid=(30.0, 40.0),
            velocity=(5.0, 5.0),
            speed=7.07,
            acceleration=(1.0, 1.0),
            direction=45.0,
            confidence=0.9
        )
        
        self.assertEqual(frame_state.frame_number, 1)
        self.assertEqual(frame_state.centroid, (30.0, 40.0))
        self.assertEqual(frame_state.speed, 7.07)
    
    def test_track_state(self):
        """Test TrackState schema."""
        frame_states = [
            FrameState(frame_number=1, bbox=[10, 10, 50, 50], centroid=(30, 30)),
            FrameState(frame_number=2, bbox=[12, 12, 52, 52], centroid=(32, 32)),
        ]
        
        track_state = TrackState(
            track_id=0,
            frame_states=frame_states,
            diagnostics=TrackDiagnostics(),
            first_seen_frame=1,
            last_seen_frame=2
        )
        
        self.assertEqual(track_state.track_id, 0)
        self.assertEqual(len(track_state.frame_states), 2)
    
    def test_cv_agent_config(self):
        """Test CVAgentConfig schema."""
        config = CVAgentConfig(
            conf_threshold=0.3,
            iou_threshold=0.4,
            smoothing_factor=0.8
        )
        
        self.assertEqual(config.conf_threshold, 0.3)
        self.assertEqual(config.iou_threshold, 0.4)
        self.assertEqual(config.smoothing_factor, 0.8)


class TestCVAgent(unittest.TestCase):
    """Test CV Agent (requires YOLO model, may skip if not available)."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip if model not available
        import os
        if not os.path.exists("models/yolov8n_vessels.pt"):
            self.skipTest("YOLO model not found")
    
    def test_cv_agent_initialization(self):
        """Test CV Agent initialization."""
        try:
            agent = CVAgent(model_path="models/yolov8n_vessels.pt")
            self.assertIsNotNone(agent.yolo_model)
        except FileNotFoundError:
            self.skipTest("YOLO model not found")
    
    def test_track_states_to_dict(self):
        """Test conversion from TrackState to dict format."""
        agent = CVAgent(model_path="models/yolov8n_vessels.pt")
        
        frame_states = [
            FrameState(
                frame_number=1,
                bbox=[10, 10, 50, 50],
                centroid=(30, 30),
                velocity=(5, 5),
                speed=7.07,
                confidence=0.9
            )
        ]
        
        track_state = TrackState(
            track_id=0,
            frame_states=frame_states,
            diagnostics=TrackDiagnostics(),
            first_seen_frame=1,
            last_seen_frame=1
        )
        
        result = agent.track_states_to_dict([track_state])
        self.assertIn(0, result)
        self.assertEqual(len(result[0]["centroids"]), 1)


class TestOrchestrator(unittest.TestCase):
    """Test Orchestrator Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        import os
        if not os.path.exists("models/yolov8n_vessels.pt"):
            self.skipTest("YOLO model not found")
    
    def test_orchestrator_initialization(self):
        """Test Orchestrator initialization."""
        try:
            orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
            self.assertIsNotNone(orchestrator.cv_agent)
        except FileNotFoundError:
            self.skipTest("YOLO model not found")
    
    def test_compute_turn_sharpness(self):
        """Test turn sharpness computation."""
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
        
        # Create track with sharp turn
        frame_states = [
            FrameState(frame_number=1, bbox=[10, 10, 50, 50], direction=0.0),
            FrameState(frame_number=2, bbox=[12, 12, 52, 52], direction=45.0),
            FrameState(frame_number=3, bbox=[14, 14, 54, 54], direction=90.0),
        ]
        
        track_state = TrackState(
            track_id=0,
            frame_states=frame_states,
            diagnostics=TrackDiagnostics(),
            first_seen_frame=1,
            last_seen_frame=3
        )
        
        sharpness = orchestrator._compute_turn_sharpness(track_state)
        self.assertGreater(sharpness, 0.0)
        self.assertLessEqual(sharpness, 1.0)
    
    def test_compute_jerkiness(self):
        """Test jerkiness computation."""
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
        
        # Create track with varying acceleration
        frame_states = [
            FrameState(frame_number=1, bbox=[10, 10, 50, 50], acceleration=(1.0, 1.0)),
            FrameState(frame_number=2, bbox=[12, 12, 52, 52], acceleration=(5.0, 5.0)),
            FrameState(frame_number=3, bbox=[14, 14, 54, 54], acceleration=(2.0, 2.0)),
        ]
        
        track_state = TrackState(
            track_id=0,
            frame_states=frame_states,
            diagnostics=TrackDiagnostics(),
            first_seen_frame=1,
            last_seen_frame=3
        )
        
        jerkiness = orchestrator._compute_jerkiness(track_state)
        self.assertGreaterEqual(jerkiness, 0.0)
        self.assertLessEqual(jerkiness, 1.0)
    
    def test_query_tracks(self):
        """Test query_tracks method."""
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
        
        # Add some mock tracks
        frame_states = [
            FrameState(frame_number=1, bbox=[10, 10, 50, 50], centroid=(30, 30))
        ]
        track_state = TrackState(
            track_id=0,
            frame_states=frame_states,
            diagnostics=TrackDiagnostics(),
            first_seen_frame=1,
            last_seen_frame=1
        )
        orchestrator.tracks[0] = track_state
        orchestrator.track_quality[0] = 0.8
        
        # Test query
        response = orchestrator.query_tracks("How many vessels are tracked?")
        self.assertIn("1", response)
        
        response = orchestrator.query_tracks("List tracked vessels")
        self.assertIn("Track 0", response)
    
    def test_query_metrics(self):
        """Test query_metrics method."""
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
        
        # Add mock event
        event = EventSummary(
            event_type="turn_sharpness",
            track_id=0,
            frame_number=10,
            severity=0.7,
            description="Sharp turn"
        )
        orchestrator.events.append(event)
        
        response = orchestrator.query_metrics("Show events")
        self.assertIn("event", response.lower())


class TestAccelerationJerk(unittest.TestCase):
    """Test acceleration and jerk calculations."""
    
    def test_acceleration_correctness(self):
        """Test that acceleration calculation is correct."""
        # Constant velocity should give zero acceleration
        prev_velocity = (10.0, 10.0)
        curr_velocity = (10.0, 10.0)
        dt = 1.0
        
        acceleration = compute_acceleration(prev_velocity, curr_velocity, dt)
        self.assertEqual(acceleration, (0.0, 0.0))
        
        # Changing velocity should give non-zero acceleration
        curr_velocity = (15.0, 15.0)
        acceleration = compute_acceleration(prev_velocity, curr_velocity, dt)
        self.assertEqual(acceleration, (5.0, 5.0))
    
    def test_jerk_calculation(self):
        """Test jerk (acceleration change) calculation."""
        orchestrator = Orchestrator(model_path="models/yolov8n_vessels.pt")
        
        # Smooth acceleration (low jerk)
        frame_states = [
            FrameState(frame_number=1, bbox=[10, 10, 50, 50], acceleration=(1.0, 1.0)),
            FrameState(frame_number=2, bbox=[12, 12, 52, 52], acceleration=(1.1, 1.1)),
            FrameState(frame_number=3, bbox=[14, 14, 54, 54], acceleration=(1.2, 1.2)),
        ]
        
        track_state = TrackState(
            track_id=0,
            frame_states=frame_states,
            diagnostics=TrackDiagnostics(),
            first_seen_frame=1,
            last_seen_frame=3
        )
        
        jerkiness = orchestrator._compute_jerkiness(track_state)
        self.assertLess(jerkiness, 0.5)  # Should be low for smooth acceleration


if __name__ == '__main__':
    unittest.main()
