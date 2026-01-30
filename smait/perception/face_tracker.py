"""
SMAIT HRI System v2.0 - Face Tracker
Multi-face tracking using MediaPipe Face Mesh with persistent IDs.
"""

import time
import threading
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2

from smait.core.config import get_config
from smait.core.events import FaceDetection, BoundingBox

# Import MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


@dataclass
class TrackedFace:
    """Internal tracking state for a face"""
    track_id: int
    bbox: BoundingBox
    landmarks: np.ndarray
    last_seen: float
    mar_history: deque  # Mouth Aspect Ratio history for legacy fallback
    confidence: float

    # Mouth landmarks for ASD
    mouth_roi: Optional[np.ndarray] = None


class FaceTracker:
    """
    Multi-face tracker using MediaPipe Face Mesh.
    Maintains persistent track IDs across frames.
    """

    # MediaPipe landmark indices for mouth
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_CORNER = 78
    RIGHT_CORNER = 308

    # Mouth ROI landmarks (outer lip contour)
    MOUTH_LANDMARKS = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 78, 61
    ]

    # Model download URL
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(self, max_faces: int = 5):
        self.config = get_config()
        self.max_faces = max_faces

        # Ensure model is downloaded
        model_path = self._ensure_model_downloaded()

        # Initialize MediaPipe Tasks API
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=self.config.vision.min_face_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Tracking state
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.next_track_id = 0
        self.lock = threading.Lock()

        # MAR history settings (for legacy fallback)
        self.mar_history_size = int(self.config.vision.target_fps * 2)  # 2 seconds

        print(f"[FACE] Tracker initialized (max_faces={max_faces})")

    def _ensure_model_downloaded(self) -> str:
        """Download the face landmarker model if not present"""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe", "models")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "face_landmarker.task")

        if not os.path.exists(model_path):
            print("[FACE] Downloading face landmarker model...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("[FACE] Model downloaded successfully")

        return model_path
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[FaceDetection], np.ndarray]:
        """
        Process a frame and return detected faces.

        Returns:
            Tuple of (list of FaceDetection, annotated frame)
        """
        timestamp = time.time()
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect faces using the new Tasks API
        results = self.face_landmarker.detect(mp_image)

        # Prepare output frame
        output_frame = frame.copy() if self.config.show_video else None

        detections: List[FaceDetection] = []
        current_face_ids = set()

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Extract landmarks as numpy array
                landmarks = np.array([
                    [lm.x * w, lm.y * h, lm.z * w]
                    for lm in face_landmarks
                ])

                # Calculate bounding box
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x1, x2 = int(max(0, min(x_coords) - 20)), int(min(w, max(x_coords) + 20))
                y1, y2 = int(max(0, min(y_coords) - 20)), int(min(h, max(y_coords) + 20))
                bbox = BoundingBox(x1, y1, x2, y2)

                # Skip small faces
                if bbox.area < self.config.vision.min_face_area:
                    continue

                # Calculate MAR
                mar = self._calculate_mar(landmarks)

                # Extract mouth ROI for ASD
                mouth_roi = self._extract_mouth_roi(frame, landmarks)

                # Match to existing track or create new
                track_id = self._match_or_create_track(bbox, landmarks, mar, timestamp)
                current_face_ids.add(track_id)

                # Update tracked face
                with self.lock:
                    if track_id in self.tracked_faces:
                        self.tracked_faces[track_id].mouth_roi = mouth_roi

                # Create detection result
                detection = FaceDetection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=1.0,  # MediaPipe doesn't give per-face confidence
                    timestamp=timestamp,
                    landmarks=landmarks,
                    mouth_roi=mouth_roi,
                    mar=mar
                )
                detections.append(detection)

                # Draw on output frame
                if output_frame is not None:
                    self._draw_face(output_frame, detection, landmarks)

        # Clean up lost tracks
        self._cleanup_lost_tracks(current_face_ids, timestamp)

        # Draw status on frame
        if output_frame is not None:
            self._draw_status(output_frame, len(detections))

        return detections, output_frame
    
    def _calculate_mar(self, landmarks: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio"""
        upper = landmarks[self.UPPER_LIP][:2]
        lower = landmarks[self.LOWER_LIP][:2]
        left = landmarks[self.LEFT_CORNER][:2]
        right = landmarks[self.RIGHT_CORNER][:2]
        
        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)
        
        return vertical / max(horizontal, 1)
    
    def _extract_mouth_roi(self, frame: np.ndarray, landmarks: np.ndarray, size: int = 96) -> np.ndarray:
        """Extract mouth region for ASD model"""
        h, w = frame.shape[:2]
        
        # Get mouth bounding box from landmarks
        mouth_points = landmarks[self.MOUTH_LANDMARKS][:, :2].astype(np.int32)
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # Add padding
        pad = int((x_max - x_min) * 0.3)
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        # Extract and resize
        mouth_crop = frame[y_min:y_max, x_min:x_max]
        if mouth_crop.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        
        mouth_roi = cv2.resize(mouth_crop, (size, size))
        return mouth_roi
    
    def _match_or_create_track(
        self,
        bbox: BoundingBox,
        landmarks: np.ndarray,
        mar: float,
        timestamp: float
    ) -> int:
        """Match detection to existing track or create new one"""
        with self.lock:
            best_match_id = None
            best_match_dist = float('inf')
            
            # Find closest existing track by center distance
            for track_id, tracked in self.tracked_faces.items():
                dist = np.sqrt(
                    (bbox.center[0] - tracked.bbox.center[0]) ** 2 +
                    (bbox.center[1] - tracked.bbox.center[1]) ** 2
                )
                # Only match if reasonably close (within 100 pixels)
                if dist < 100 and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                tracked = self.tracked_faces[best_match_id]
                tracked.bbox = bbox
                tracked.landmarks = landmarks
                tracked.last_seen = timestamp
                tracked.confidence = 1.0
                tracked.mar_history.append({'time': timestamp, 'mar': mar})
                return best_match_id
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                mar_history = deque(maxlen=self.mar_history_size)
                mar_history.append({'time': timestamp, 'mar': mar})
                
                self.tracked_faces[track_id] = TrackedFace(
                    track_id=track_id,
                    bbox=bbox,
                    landmarks=landmarks,
                    last_seen=timestamp,
                    mar_history=mar_history,
                    confidence=1.0
                )
                return track_id
    
    def _cleanup_lost_tracks(self, current_ids: set, timestamp: float):
        """Remove tracks that haven't been seen recently"""
        grace_period = self.config.session.face_lost_grace_seconds
        
        with self.lock:
            lost_ids = []
            for track_id, tracked in self.tracked_faces.items():
                if track_id not in current_ids:
                    if timestamp - tracked.last_seen > grace_period:
                        lost_ids.append(track_id)
            
            for track_id in lost_ids:
                del self.tracked_faces[track_id]
                if self.config.debug:
                    print(f"[FACE] Lost track {track_id}")
    
    def _draw_face(self, frame: np.ndarray, detection: FaceDetection, landmarks: np.ndarray):
        """Draw face detection on frame"""
        bbox = detection.bbox

        # Draw bounding box
        cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)

        # Draw track ID
        cv2.putText(
            frame,
            f"ID: {detection.track_id}",
            (bbox.x1, bbox.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Draw MAR
        cv2.putText(
            frame,
            f"MAR: {detection.mar:.3f}",
            (bbox.x1, bbox.y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Draw lip landmarks as connected contour
        lip_pts = [tuple(landmarks[idx][:2].astype(int)) for idx in self.MOUTH_LANDMARKS]
        for i in range(len(lip_pts) - 1):
            cv2.line(frame, lip_pts[i], lip_pts[i + 1], (0, 255, 0), 1)

        # Draw key points larger: corners and center
        for idx in [self.UPPER_LIP, self.LOWER_LIP, self.LEFT_CORNER, self.RIGHT_CORNER]:
            pt = tuple(landmarks[idx][:2].astype(int))
            cv2.circle(frame, pt, 3, (0, 255, 255), -1)  # Yellow for key points
    
    def _draw_status(self, frame: np.ndarray, num_faces: int):
        """Draw status information on frame"""
        h = frame.shape[0]
        
        status = f"Faces: {num_faces}"
        color = (0, 255, 0) if num_faces > 0 else (0, 0, 255)
        
        cv2.putText(
            frame,
            status,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
    def get_mar_history(self, track_id: int, seconds: float = 1.5) -> List[Dict]:
        """Get MAR history for a specific track"""
        with self.lock:
            if track_id not in self.tracked_faces:
                return []
            
            tracked = self.tracked_faces[track_id]
            cutoff = time.time() - seconds
            
            return [
                entry for entry in tracked.mar_history
                if entry['time'] >= cutoff
            ]
    
    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs"""
        with self.lock:
            return list(self.tracked_faces.keys())
    
    def cleanup(self):
        """Release resources"""
        try:
            self.face_mesh.close()
        except:
            pass
