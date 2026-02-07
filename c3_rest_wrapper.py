"""
C3.ai REST Wrapper for NAVIS
Provides simple REST endpoints for C3.ai integration:
- /health: Health check
- /infer: Inference endpoint (accepts video path or frame data)
- /metrics: System metrics
"""

import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from datetime import datetime
import json

from motion_core import detect_vessels_with_yolo
from orchestrator import Orchestrator


class C3RESTWrapper:
    """REST wrapper for C3.ai integration."""
    
    def __init__(self, model_path: str = "models/yolov8n_vessels.pt"):
        """Initialize wrapper with YOLO model."""
        self.model_path = Path(model_path)
        self.orchestrator = None
        self.yolo_model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load YOLO model."""
        if self.yolo_model is None:
            from ultralytics import YOLO
            import torch
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.yolo_model = YOLO(str(self.model_path))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model.to(device)
            self.device = device
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "NAVIS",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": self.yolo_model is not None,
            "device": getattr(self, 'device', 'unknown')
        }
    
    def infer_from_video_path(self, video_path: str, frame_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to video file (relative to /app/media or absolute)
            frame_limit: Optional limit on number of frames to process
            
        Returns:
            Dictionary with detections and metadata
        """
        # Resolve video path
        video_path_obj = Path(video_path)
        
        # Check if absolute path exists
        if not video_path_obj.is_absolute():
            # Try relative to media directory
            media_path = Path("/app/media") / video_path_obj
            if media_path.exists():
                video_path_obj = media_path
            elif video_path_obj.exists():
                pass  # Use as-is
            else:
                raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        elif not video_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail=f"Could not open video: {video_path}")
        
        try:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process frames
            frame_count = 0
            all_detections = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                detections = detect_vessels_with_yolo(frame, self.yolo_model, conf_threshold=0.25)
                
                frame_detections = []
                for det in detections:
                    x1, y1, x2, y2, conf = det[:5]
                    centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    area = (x2 - x1) * (y2 - y1)
                    
                    frame_detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "centroid": [float(centroid[0]), float(centroid[1])],
                        "area": float(area)
                    })
                
                all_detections.append({
                    "frame": frame_count,
                    "detections": frame_detections,
                    "detection_count": len(frame_detections)
                })
                
                frame_count += 1
                
                if frame_limit and frame_count >= frame_limit:
                    break
            
            cap.release()
            
            return {
                "video_path": str(video_path_obj),
                "fps": fps,
                "resolution": [width, height],
                "total_frames": total_frames,
                "processed_frames": frame_count,
                "detections": all_detections,
                "summary": {
                    "total_detections": sum(len(f["detections"]) for f in all_detections),
                    "frames_with_detections": sum(1 for f in all_detections if f["detection_count"] > 0),
                    "avg_detections_per_frame": sum(f["detection_count"] for f in all_detections) / max(frame_count, 1)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            cap.release()
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    def infer_from_frame_data(self, frame_data: str, format: str = "base64") -> Dict[str, Any]:
        """
        Run inference on a single frame.
        
        Args:
            frame_data: Base64-encoded image or image data
            format: "base64" or "raw"
            
        Returns:
            Dictionary with detections
        """
        try:
            # Decode frame
            if format == "base64":
                image_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Assume raw bytes
                if isinstance(frame_data, str):
                    image_bytes = frame_data.encode('latin-1')
                else:
                    image_bytes = frame_data
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode image")
            
            # Run detection
            detections = detect_vessels_with_yolo(frame, self.yolo_model, conf_threshold=0.25)
            
            frame_detections = []
            for det in detections:
                x1, y1, x2, y2, conf = det[:5]
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                area = (x2 - x1) * (y2 - y1)
                
                frame_detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "area": float(area)
                })
            
            return {
                "detections": frame_detections,
                "detection_count": len(frame_detections),
                "frame_shape": list(frame.shape),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Frame processing error: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            import psutil
        except ImportError:
            psutil = None
        
        import torch
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "device": getattr(self, 'device', 'unknown'),
                "model_path": str(self.model_path),
                "model_loaded": self.yolo_model is not None
            }
        }
        
        # Add system metrics if psutil is available
        if psutil:
            metrics["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024)
            }
        else:
            metrics["system"] = {"note": "psutil not available"}
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024)
            }
        else:
            metrics["gpu"] = {"available": False}
        
        return metrics


# Global wrapper instance
_c3_wrapper: Optional[C3RESTWrapper] = None


def get_c3_wrapper() -> C3RESTWrapper:
    """Get or create C3 wrapper instance."""
    global _c3_wrapper
    if _c3_wrapper is None:
        _c3_wrapper = C3RESTWrapper()
    return _c3_wrapper
