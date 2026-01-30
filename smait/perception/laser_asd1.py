"""
SMAIT HRI System v2.0 - LASER Active Speaker Detection

LASER (Lip landmark Assisted Speaker dEtection for Robustness) uses explicit
lip landmark guidance to improve active speaker detection accuracy.

Key advantages over MAR heuristic and Light-ASD:
- Uses existing MediaPipe landmarks (no additional compute)
- Robust to audio-visual desynchronization (+3-4% mAP)
- Handles visually crowded scenes better
- Works without landmarks at inference (consistency loss)

Paper: "LASER: Lip Landmark Assisted Speaker Detection for Robustness"
GitHub: https://github.com/plnguyen2908/LASER_ASD
"""

import time
import threading
from typing import Optional, List, Dict, Tuple
from collections import deque
import numpy as np

from smait.core.config import get_config
from smait.core.events import FaceDetection, ActiveSpeakerResult


# MediaPipe lip landmark indices (subset of 468 face mesh landmarks)
LIP_LANDMARKS = [
    # Outer lip contour
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Inner lip contour  
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Key points for MAR
    13,  # Upper lip center
    14,  # Lower lip center
    78,  # Left corner
    308, # Right corner
]

# Indices for extracting 20 key lip landmarks used by LASER
LASER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # Outer upper + lower
    78, 95, 88, 178, 87, 14, 317, 402, 318, 308,   # Inner + corners
]


class LASERBackend:
    """
    LASER Active Speaker Detection backend.
    
    Uses lip landmarks from MediaPipe to guide attention to speech-relevant
    regions. Can operate in two modes:
    
    1. Full mode: Uses ONNX model with landmark encoding
    2. Lite mode: Uses landmark-based heuristics (no model needed)
    
    The lite mode is a reasonable approximation when the full model
    isn't available, leveraging the same lip landmark principles.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = get_config()
        self.model_path = model_path
        
        # ONNX session for full LASER model
        self.session = None
        self._use_lite_mode = True
        
        # Per-face tracking for temporal analysis
        self.lip_histories: Dict[int, deque] = {}
        self.audio_histories: Dict[int, deque] = {}
        self._head_histories: Dict[int, deque] = {}  # For head motion compensation
        
        # History settings
        self.history_size = int(self.config.vision.target_fps * 2)  # 2 seconds
        self.lock = threading.Lock()
        
        # Thresholds (tuned for intrinsic mouth metrics)
        self.min_lip_movement = 0.008       # Minimum MAR change to count as movement
        self.min_mar_for_speech = 0.05      # Minimum MAR value (mouth must be slightly open)
        self.head_motion_threshold = 0.02   # Above this, reduce confidence due to head motion
        self.av_sync_threshold = 0.3        # Audio-visual sync correlation threshold
        self.temporal_window_ms = 500       # Window for temporal analysis
        
        # Try to load full model
        self._load_model()
        
        mode = "full model" if self.session else "lite (landmark heuristics)"
        print(f"[ASD] LASER backend initialized ({mode})")
    
    def _load_model(self):
        """Try to load the LASER ONNX model"""
        if self.model_path is None:
            # Try default locations
            import os
            possible_paths = [
                "models/laser_asd.onnx",
                os.path.expanduser("~/.smait/models/laser_asd.onnx"),
                "/opt/smait/models/laser_asd.onnx"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
        
        if self.model_path:
            try:
                import onnxruntime as ort
                
                # Try GPU first, fall back to CPU
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=providers
                )
                self._use_lite_mode = False
                print(f"[ASD] LASER model loaded from {self.model_path}")
                
            except Exception as e:
                print(f"[ASD] Failed to load LASER model: {e}")
                print(f"[ASD] Using lite mode (landmark heuristics)")
                self._use_lite_mode = True
        else:
            print(f"[ASD] LASER model not found, using lite mode")
            print(f"[ASD] To use full model, download to ~/.smait/models/laser_asd.onnx")
    
    def _extract_lip_landmarks(self, face: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract and normalize lip landmarks from face detection.
        
        Returns:
            Normalized lip landmarks (20, 2) or None if not available
        """
        if face.landmarks is None:
            return None
        
        try:
            # Extract lip landmarks
            lip_points = face.landmarks[LASER_LIP_INDICES, :2]  # (20, 2) x,y only
            
            # Normalize to face bounding box
            bbox = face.bbox
            center_x = (bbox.x1 + bbox.x2) / 2
            center_y = (bbox.y1 + bbox.y2) / 2
            scale = max(bbox.width, bbox.height)
            
            # Center and scale
            normalized = (lip_points - np.array([center_x, center_y])) / scale
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            if self.config.debug:
                print(f"[ASD] Lip landmark extraction error: {e}")
            return None
    
    def _compute_lip_movement(self, track_id: int, current_landmarks: np.ndarray, timestamp: float) -> float:
        """
        Compute lip movement score based on INTRINSIC mouth shape changes.
        
        Uses head-pose-invariant metrics:
        1. Mouth Aspect Ratio (MAR) - vertical/horizontal ratio
        2. Inner lip distances - pairwise distances between lip points
        
        These metrics are invariant to head translation and rotation because
        they measure relative distances within the mouth, not absolute positions.
        """
        # Compute current intrinsic mouth metrics
        mouth_metrics = self._compute_mouth_metrics(current_landmarks)
        
        with self.lock:
            if track_id not in self.lip_histories:
                self.lip_histories[track_id] = deque(maxlen=self.history_size)
            
            history = self.lip_histories[track_id]
            
            # Store metrics (not raw landmarks)
            history.append({
                'time': timestamp,
                'metrics': mouth_metrics
            })
            
            if len(history) < 3:
                return 0.0
        
        # Get recent history (last 500ms)
        cutoff = timestamp - (self.temporal_window_ms / 1000.0)
        recent = [h for h in history if h['time'] >= cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        # Compute movement as change in intrinsic metrics
        total_movement = 0.0
        for i in range(1, len(recent)):
            prev_metrics = recent[i-1]['metrics']
            curr_metrics = recent[i]['metrics']
            
            # MAR change (primary signal for speech)
            mar_change = abs(curr_metrics['mar'] - prev_metrics['mar'])
            
            # Inner mouth area change
            area_change = abs(curr_metrics['inner_area'] - prev_metrics['inner_area'])
            
            # Lip spread change (mouth width variation)
            spread_change = abs(curr_metrics['spread'] - prev_metrics['spread'])
            
            # Weighted combination - MAR is most important for speech
            frame_movement = (
                0.6 * mar_change +           # Mouth opening/closing
                0.25 * area_change +         # Inner mouth area
                0.15 * spread_change         # Lip spreading
            )
            total_movement += frame_movement
        
        # Normalize by number of frames
        avg_movement = total_movement / (len(recent) - 1)
        
        return avg_movement
    
    def _compute_mouth_metrics(self, landmarks: np.ndarray) -> dict:
        """
        Compute head-pose-invariant mouth shape metrics.
        
        These use DISTANCES BETWEEN LANDMARKS, not absolute positions,
        making them invariant to head translation and rotation.
        
        Args:
            landmarks: Normalized lip landmarks (20, 2)
        
        Returns:
            Dict with intrinsic mouth metrics
        """
        # Landmark indices in our 20-point lip set:
        # Outer: 0-9, Inner: 10-19
        # Key points (approximate mapping):
        # - Upper lip center: ~index 5 (outer), ~15 (inner) 
        # - Lower lip center: ~index 5 (outer), ~15 (inner)
        # - Left corner: ~index 0
        # - Right corner: ~index 5
        
        try:
            # Get key points for metric computation
            # Using the 20-point LASER lip landmark set
            left_corner = landmarks[0]      # Outer left
            right_corner = landmarks[5]     # Outer right (approximate)
            upper_outer = landmarks[3]      # Upper lip outer center
            lower_outer = landmarks[8]      # Lower lip outer center
            upper_inner = landmarks[13]     # Upper lip inner
            lower_inner = landmarks[18]     # Lower lip inner
            
            # 1. Mouth width (horizontal distance between corners)
            mouth_width = np.linalg.norm(right_corner - left_corner)
            
            # 2. Vertical opening (distance between upper and lower lip)
            vertical_outer = np.linalg.norm(upper_outer - lower_outer)
            vertical_inner = np.linalg.norm(upper_inner - lower_inner)
            
            # 3. MAR (Mouth Aspect Ratio) - normalized by width
            # This is invariant to scale AND head pose
            mar = vertical_inner / max(mouth_width, 0.001)
            
            # 4. Inner mouth area (approximate using inner lip points)
            # Use subset of inner lip points to estimate area
            inner_points = landmarks[10:20]  # Inner lip contour
            inner_area = self._polygon_area(inner_points)
            
            # Normalize area by mouth width squared (scale invariant)
            inner_area_normalized = inner_area / max(mouth_width ** 2, 0.001)
            
            # 5. Lip spread ratio
            spread = mouth_width / max(vertical_outer + 0.001, 0.001)
            
            return {
                'mar': mar,
                'inner_area': inner_area_normalized,
                'spread': spread,
                'width': mouth_width,
                'vertical': vertical_inner
            }
            
        except (IndexError, ValueError) as e:
            # Fallback to safe defaults
            return {
                'mar': 0.0,
                'inner_area': 0.0,
                'spread': 1.0,
                'width': 0.1,
                'vertical': 0.0
            }
    
    def _polygon_area(self, points: np.ndarray) -> float:
        """
        Compute area of polygon using shoelace formula.
        Works for any head pose since it uses relative point positions.
        """
        n = len(points)
        if n < 3:
            return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i, 0] * points[j, 1]
            area -= points[j, 0] * points[i, 1]
        
        return abs(area) / 2.0
    
    def _compute_vertical_movement(self, track_id: int, face: FaceDetection) -> float:
        """
        Compute vertical lip movement velocity (mouth opening/closing rate).
        
        Uses MAR velocity which is inherently head-pose-invariant.
        """
        if face.landmarks is None:
            return 0.0
        
        # Compute current MAR from full face landmarks
        mar = self._compute_mar_from_face(face)
        
        # Get MAR history from lip_histories
        with self.lock:
            if track_id not in self.lip_histories:
                return 0.0
            
            history = list(self.lip_histories[track_id])
        
        if len(history) < 3:
            return 0.0
        
        # Compute MAR velocity from recent history
        mar_values = []
        times = []
        
        for h in history[-10:]:
            if 'metrics' in h:
                mar_values.append(h['metrics']['mar'])
                times.append(h['time'])
        
        if len(mar_values) < 2:
            return 0.0
        
        # Compute velocity (rate of MAR change)
        velocities = []
        for i in range(1, len(mar_values)):
            dt = times[i] - times[i-1]
            if dt > 0:
                vel = abs(mar_values[i] - mar_values[i-1]) / dt
                velocities.append(vel)
        
        return np.mean(velocities) if velocities else 0.0
    
    def _compute_mar_from_face(self, face: FaceDetection) -> float:
        """Compute MAR from full MediaPipe face landmarks"""
        if face.landmarks is None:
            return 0.0
        
        try:
            # MediaPipe landmark indices for lips
            # Upper lip: 13, Lower lip: 14, Corners: 78, 308
            upper = face.landmarks[13, :2]
            lower = face.landmarks[14, :2]
            left = face.landmarks[78, :2]
            right = face.landmarks[308, :2]
            
            vertical = np.linalg.norm(upper - lower)
            horizontal = np.linalg.norm(left - right)
            
            return vertical / max(horizontal, 0.001)
        except:
            return face.mar if hasattr(face, 'mar') else 0.0
    
    def _compute_head_motion(self, face: FaceDetection, track_id: int) -> float:
        """
        Compute head motion using non-lip face landmarks.
        
        Compares movement of stable face points (eyes, nose) to detect
        when head is moving vs when only mouth is moving.
        """
        if face.landmarks is None:
            return 0.0
        
        # Use stable landmarks: nose tip (1), eye corners (33, 263)
        try:
            stable_points = face.landmarks[[1, 33, 263], :2]  # nose, left eye, right eye
        except:
            return 0.0
        
        with self.lock:
            if track_id not in self._head_histories:
                self._head_histories[track_id] = deque(maxlen=30)
            
            history = self._head_histories[track_id]
            history.append({
                'time': face.timestamp,
                'points': stable_points.copy()
            })
            
            if len(history) < 2:
                return 0.0
        
        # Compute displacement of stable points
        prev = history[-2]['points']
        curr = history[-1]['points']
        
        displacement = np.mean(np.linalg.norm(curr - prev, axis=1))
        
        return displacement
    
    def _compute_av_sync_score(
        self,
        track_id: int,
        lip_movement: float,
        audio: Optional[np.ndarray]
    ) -> float:
        """
        Compute audio-visual synchronization score.
        
        High score means lip movement correlates with audio energy.
        """
        if audio is None or len(audio) == 0:
            # No audio available, rely on visual only
            return 0.5
        
        # Compute audio energy
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio
        
        audio_energy = np.sqrt(np.mean(audio_float ** 2))
        
        # Simple correlation: both high or both low is good sync
        # Normalize to [0, 1]
        lip_norm = min(lip_movement / 0.1, 1.0)  # 0.1 is high movement
        audio_norm = min(audio_energy / 0.1, 1.0)  # 0.1 is loud audio
        
        # Sync score: high when both are similar
        sync = 1.0 - abs(lip_norm - audio_norm)
        
        return sync
    
    def detect(
        self,
        face: FaceDetection,
        audio: Optional[np.ndarray] = None
    ) -> ActiveSpeakerResult:
        """
        Detect if the face is speaking using LASER approach.
        
        Uses lip landmarks to guide detection, with optional audio correlation.
        """
        timestamp = face.timestamp
        
        # Extract lip landmarks
        lip_landmarks = self._extract_lip_landmarks(face)
        
        if lip_landmarks is None:
            # No landmarks, can't do LASER detection
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=False,
                probability=0.0,
                timestamp=timestamp,
                lip_movement=0.0
            )
        
        # Use full model if available
        if self.session and not self._use_lite_mode:
            return self._detect_with_model(face, lip_landmarks, audio)
        
        # Lite mode: landmark-based heuristics
        return self._detect_lite(face, lip_landmarks, audio)
    
    def _detect_with_model(
        self,
        face: FaceDetection,
        lip_landmarks: np.ndarray,
        audio: Optional[np.ndarray]
    ) -> ActiveSpeakerResult:
        """Full LASER model inference"""
        try:
            # Prepare inputs for LASER model
            # Model expects: lip_landmarks (B, T, 20, 2), audio_features (B, T, 13)
            
            # For single frame, T=1
            lip_input = lip_landmarks[np.newaxis, np.newaxis, :, :]  # (1, 1, 20, 2)
            
            # Extract audio features (MFCCs)
            if audio is not None:
                audio_features = self._extract_audio_features(audio)
            else:
                audio_features = np.zeros((1, 1, 13), dtype=np.float32)
            
            # Run inference
            inputs = {
                'lip_landmarks': lip_input,
                'audio_features': audio_features
            }
            
            outputs = self.session.run(None, inputs)
            probability = float(outputs[0][0])
            
            # Compute lip movement for debugging
            lip_movement = self._compute_lip_movement(
                face.track_id, lip_landmarks, face.timestamp
            )
            
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=probability > self.config.vision.asd_threshold,
                probability=probability,
                timestamp=face.timestamp,
                lip_movement=lip_movement,
                audio_visual_sync=probability
            )
            
        except Exception as e:
            if self.config.debug:
                print(f"[ASD] LASER model error: {e}")
            # Fall back to lite mode
            return self._detect_lite(face, lip_landmarks, audio)
    
    def _detect_lite(
        self,
        face: FaceDetection,
        lip_landmarks: np.ndarray,
        audio: Optional[np.ndarray]
    ) -> ActiveSpeakerResult:
        """
        LASER lite mode: Use head-pose-invariant mouth metrics.
        
        Key improvements over simple landmark displacement:
        1. Uses MAR (Mouth Aspect Ratio) which is pose-invariant
        2. Tracks intrinsic mouth shape changes (area, spread)
        3. Compensates for head motion using stable face landmarks
        4. Requires minimum mouth opening to avoid false positives from smiling
        """
        timestamp = face.timestamp
        
        # Compute intrinsic lip movement metrics
        lip_movement = self._compute_lip_movement(face.track_id, lip_landmarks, timestamp)
        mar_velocity = self._compute_vertical_movement(face.track_id, face)
        current_mar = self._compute_mar_from_face(face)
        
        # Compute head motion to compensate for head movement
        head_motion = self._compute_head_motion(face, face.track_id)
        
        # Compute audio-visual sync
        av_sync = self._compute_av_sync_score(face.track_id, lip_movement, audio)
        
        # === Decision Logic ===
        
        # 1. Check if mouth is open enough to be speaking
        mouth_open_enough = current_mar > self.min_mar_for_speech
        
        # 2. Check if there's significant lip movement (MAR changes)
        significant_movement = lip_movement > self.min_lip_movement
        
        # 3. Check if head is relatively still (or compensate)
        head_is_still = head_motion < self.head_motion_threshold
        
        # 4. Compute probability based on metrics
        # Weight the different signals
        if mouth_open_enough and significant_movement:
            # Base probability from lip movement
            movement_score = min(lip_movement / 0.03, 1.0)  # 0.03 is strong movement
            mar_score = min(mar_velocity / 0.3, 1.0)        # 0.3 is fast MAR change
            
            # Combine scores
            base_probability = 0.5 * movement_score + 0.35 * mar_score + 0.15 * av_sync
            
            # Penalize if head is moving a lot (less confident)
            if not head_is_still:
                # Reduce confidence when head is moving
                # The more head motion, the less confident we are about lip detection
                head_penalty = min(head_motion / self.head_motion_threshold, 2.0) - 1.0
                head_penalty = max(0, head_penalty) * 0.4  # Max 40% reduction
                base_probability *= (1.0 - head_penalty)
            
            probability = base_probability
        else:
            # Mouth not open or no movement
            probability = 0.1 * av_sync  # Low probability, only audio correlation
        
        # Apply threshold
        is_speaking = probability > self.config.vision.asd_threshold and mouth_open_enough
        
        # Debug output
        if self.config.debug and face.track_id == 0:  # Only for first face
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 30 == 0:  # Print every 30 frames
                print(f"[ASD-DEBUG] MAR={current_mar:.3f} movement={lip_movement:.4f} "
                      f"head={head_motion:.4f} prob={probability:.2f} speaking={is_speaking}")
        
        return ActiveSpeakerResult(
            track_id=face.track_id,
            is_speaking=is_speaking,
            probability=probability,
            timestamp=timestamp,
            lip_movement=lip_movement,
            audio_visual_sync=av_sync
        )
    
    def _extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features for LASER model"""
        try:
            import librosa
            
            if audio.dtype == np.int16:
                audio_float = audio.astype(np.float32) / 32768.0
            else:
                audio_float = audio
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio_float,
                sr=16000,
                n_mfcc=13,
                n_fft=512,
                hop_length=160
            )
            
            # Average over time for single frame
            mfcc_avg = np.mean(mfcc, axis=1, keepdims=True)
            
            # Shape: (1, 1, 13) for batch, time, features
            return mfcc_avg.T[np.newaxis, :, :].astype(np.float32)
            
        except ImportError:
            return np.zeros((1, 1, 13), dtype=np.float32)
    
    def detect_batch(
        self,
        faces: List[FaceDetection],
        audio: Optional[np.ndarray] = None
    ) -> List[ActiveSpeakerResult]:
        """Detect speaking status for multiple faces"""
        return [self.detect(face, audio) for face in faces]
    
    def cleanup_track(self, track_id: int):
        """Clean up tracking data for a lost face"""
        with self.lock:
            if track_id in self.lip_histories:
                del self.lip_histories[track_id]
            if track_id in self.audio_histories:
                del self.audio_histories[track_id]
            if track_id in self._head_histories:
                del self._head_histories[track_id]
    
    @property
    def using_model(self) -> bool:
        """Whether using full LASER model or lite mode"""
        return not self._use_lite_mode


def create_laser_backend(config=None) -> LASERBackend:
    """Factory function to create LASER backend"""
    if config is None:
        config = get_config()
    
    return LASERBackend(model_path=config.vision.asd_model_path)
