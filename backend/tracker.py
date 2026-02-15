"""
Object tracking module using YOLO.

Improvements:
- Better error handling for camera initialization
- Frame validation to prevent crashes
- Configurable tracker name from Config
- More robust camera handling
"""

import cv2
import time
from ultralytics import YOLO
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class ObjectTracker:
    """Handles YOLO object detection and tracking"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the object tracker
        
        Args:
            model_path: Path to YOLO model file
        """
        self.model_path = model_path or Config.YOLO_MODEL
        self.model = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, track: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process a frame with YOLO tracking
        
        Args:
            frame: Input frame (BGR format)
            track: Whether to use tracking (True) or just detection (False)
            
        Returns:
            Tuple of (annotated_frame, metadata)
        """
        # Validate frame
        if frame is None or frame.size == 0:
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), {
                "num_detections": 0,
                "detections": [],
                "detected_classes": [],
                "fps": round(self.fps, 1),
                "frame_count": self.frame_count,
                "error": "Invalid frame"
            }
        
        # Validate frame dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"⚠️ Warning: Unexpected frame shape: {frame.shape}")
            return frame, {
                "num_detections": 0,
                "detections": [],
                "detected_classes": [],
                "fps": round(self.fps, 1),
                "frame_count": self.frame_count,
                "error": "Invalid frame dimensions"
            }
        
        self.frame_count += 1
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / (current_time - self.start_time)
            self.last_fps_update = current_time
        
        # Get tracker config
        tracker_config = getattr(Config, "TRACKER_CONFIG", "bytetrack.yaml")
        
        # Run YOLO inference
        try:
            if track:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=tracker_config,
                    verbose=False,
                    conf=Config.CONFIDENCE_THRESHOLD,
                    classes=Config.ALLOWED_CLASSES
                )
            else:
                results = self.model(
                    frame,
                    verbose=False,
                    conf=Config.CONFIDENCE_THRESHOLD,
                    classes=Config.ALLOWED_CLASSES
                )
        except Exception as e:
            print(f"✗ YOLO inference error: {e}")
            return frame, {
                "num_detections": 0,
                "detections": [],
                "detected_classes": [],
                "fps": round(self.fps, 1),
                "frame_count": self.frame_count,
                "error": str(e)
            }
        
        # Extract metadata
        metadata = self._extract_metadata(results[0])
        
        # Create custom annotated frame with only tracking IDs
        annotated_frame = self._draw_tracking_ids_only(frame.copy(), results[0])
        
        return annotated_frame, metadata
    
    def _draw_tracking_ids_only(self, frame: np.ndarray, result) -> np.ndarray:
        """
        Draw bounding boxes with only tracking IDs (no class names)
        
        Args:
            frame: Input frame to annotate
            result: YOLO result object
            
        Returns:
            Annotated frame with tracking IDs only
        """
        if result.boxes is None or len(result.boxes) == 0:
            return frame
        
        boxes = result.boxes
        
        # Extract data
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            return frame  # No tracking IDs, return original frame
        
        bboxes = boxes.xyxy.cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()
        
        # Draw each detection
        for i, (bbox, track_id, conf) in enumerate(zip(bboxes, track_ids, confidences)):
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            color = self._get_color_for_id(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with only tracking ID
            label = f"ID: {track_id}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width + 5, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, 
                       (x1 + 2, y1 - baseline - 2), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color for a tracking ID
        
        Args:
            track_id: Tracking ID
            
        Returns:
            BGR color tuple
        """
        # Use a simple hash to generate consistent colors
        np.random.seed(int(track_id))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        np.random.seed()  # Reset seed
        return color
    
    def _extract_metadata(self, result) -> Dict:
        """
        Extract detection metadata from YOLO result
        
        Args:
            result: YOLO result object
            
        Returns:
            Dictionary containing detection metadata
        """
        detections = []
        detected_classes = set()
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            # Extract Track IDs (Handle cases where no ID is assigned yet)
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().tolist()
            else:
                track_ids = [None] * len(boxes)
                
            class_ids = boxes.cls.cpu().numpy().tolist()
            confidences = boxes.conf.cpu().numpy().tolist()
            bboxes = boxes.xyxy.cpu().numpy().tolist()  # [x1, y1, x2, y2]
            
            for i, class_id in enumerate(class_ids):
                class_name = self.model.names[int(class_id)]
                detected_classes.add(class_name)
                
                detection = {
                    "track_id": int(track_ids[i]) if track_ids[i] is not None else None,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": round(confidences[i], 3),
                    "bbox": [round(coord, 2) for coord in bboxes[i]]
                }
                detections.append(detection)
        
        metadata = {
            "num_detections": len(detections),
            "detections": detections,
            "detected_classes": list(sorted(detected_classes)),
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count
        }
        
        return metadata
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def reset_stats(self):
        """Reset frame count and timing stats"""
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0


class CameraCapture:
    """Handles camera capture with configuration"""
    
    def __init__(self, camera_index: int = None, width: int = None, height: int = None):
        """
        Initialize camera capture
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index or Config.CAMERA_INDEX
        self.width = width or Config.CAMERA_WIDTH
        self.height = height or Config.CAMERA_HEIGHT
        self.cap = None
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with settings"""
        try:
            # Try to open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.camera_index}. "
                    "Please check if camera is connected and not in use by another application."
                )
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Try to set higher FPS if possible
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"✓ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Warn if resolution doesn't match
            if actual_width != self.width or actual_height != self.height:
                print(f"⚠️ Warning: Requested {self.width}x{self.height} but got {actual_width}x{actual_height}")
            
        except Exception as e:
            print(f"✗ Error initializing camera: {e}")
            raise
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        try:
            success, frame = self.cap.read()
            return success, frame
        except Exception as e:
            print(f"✗ Error reading frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("✓ Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get camera properties.
        
        Returns:
            Dictionary with camera properties
        """
        if not self.is_opened():
            return {}
        
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "backend": self.cap.getBackendName()
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()
