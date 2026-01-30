"""
SMAIT HRI System v2.0 - Active Speaker Detection (ASD)
Determines which face is currently speaking.

Supports (in order of recommendation):
- LASER (landmark-assisted, uses MediaPipe, RECOMMENDED)
- Light-ASD (deep learning, separate model)
- MAR Heuristic (geometric, fallback)
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Tuple
from collections import deque
import numpy as np

from smait.core.config import get_config, SpeakerDetectionBackend
from smait.core.events import FaceDetection, ActiveSpeakerResult


class ASDBackend(ABC):
    """Abstract base class for ASD backends"""
    
    @abstractmethod
    def detect(
        self,
        face: FaceDetection,
        audio: Optional[np.ndarray] = None
    ) -> ActiveSpeakerResult:
        """Detect if the face is speaking"""
        pass
    
    @abstractmethod
    def detect_batch(
        self,
        faces: List[FaceDetection],
        audio: Optional[np.ndarray] = None
    ) -> List[ActiveSpeakerResult]:
        """Detect speaking status for multiple faces"""
        pass


class MARHeuristicASD(ASDBackend):
    """
    MAR-based heuristic for active speaker detection.
    This is the fallback method (same as v1.0 but improved).
    
    Detects speaking based on:
    - Standard deviation of MAR over time
    - Range of MAR values
    - Velocity of MAR changes
    
    v2.0 Improvement: Motion compensation
    - Ignores MAR changes when head is moving
    - Prevents false positives from head nods/shakes
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Thresholds from config
        self.std_threshold = self.config.vision.mar_movement_std
        self.range_threshold = self.config.vision.mar_movement_range
        self.velocity_threshold = self.config.vision.mar_movement_velocity
        
        # Per-face history
        self.mar_histories: Dict[int, deque] = {}
        self.position_histories: Dict[int, deque] = {}  # Track face position for motion detection
        self.history_size = int(self.config.vision.target_fps * 2)  # 2 seconds
        
        # Motion thresholds (pixels per second)
        self.head_motion_threshold = 50.0  # If face moves faster than this, ignore MAR
        self.stability_window = 0.3  # Seconds of stability required
        
        self.lock = threading.Lock()
        
        print("[ASD] MAR Heuristic backend initialized (with motion compensation)")
    
    def update_tracking(self, track_id: int, mar: float, face_center: Tuple[int, int], timestamp: float):
        """Update MAR and position history for a track"""
        with self.lock:
            # Initialize histories if needed
            if track_id not in self.mar_histories:
                self.mar_histories[track_id] = deque(maxlen=self.history_size)
            if track_id not in self.position_histories:
                self.position_histories[track_id] = deque(maxlen=self.history_size)
            
            self.mar_histories[track_id].append({
                'time': timestamp,
                'mar': mar
            })
            
            self.position_histories[track_id].append({
                'time': timestamp,
                'x': face_center[0],
                'y': face_center[1]
            })
    
    def _calculate_head_motion(self, track_id: int, timestamp: float) -> float:
        """Calculate head motion speed (pixels per second)"""
        with self.lock:
            if track_id not in self.position_histories:
                return 0.0
            
            history = list(self.position_histories[track_id])
        
        if len(history) < 2:
            return 0.0
        
        # Look at recent positions (last 0.3 seconds)
        cutoff = timestamp - self.stability_window
        recent = [h for h in history if h['time'] >= cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(recent)):
            dx = recent[i]['x'] - recent[i-1]['x']
            dy = recent[i]['y'] - recent[i-1]['y']
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # Calculate time span
        time_span = recent[-1]['time'] - recent[0]['time']
        if time_span <= 0:
            return 0.0
        
        # Motion speed in pixels per second
        return total_distance / time_span
    
    def _is_head_stable(self, track_id: int, timestamp: float) -> bool:
        """Check if head is stable enough to trust MAR readings"""
        motion_speed = self._calculate_head_motion(track_id, timestamp)
        return motion_speed < self.head_motion_threshold
    
    def detect(
        self,
        face: FaceDetection,
        audio: Optional[np.ndarray] = None
    ) -> ActiveSpeakerResult:
        """Detect if the face is speaking using MAR heuristic with motion compensation"""
        
        timestamp = face.timestamp
        
        # Update tracking history
        self.update_tracking(face.track_id, face.mar, face.bbox.center, timestamp)
        
        # Check if head is stable
        head_stable = self._is_head_stable(face.track_id, timestamp)
        head_motion = self._calculate_head_motion(face.track_id, timestamp)
        
        # If head is moving too much, don't count as speaking
        if not head_stable:
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=False,
                probability=0.0,
                timestamp=timestamp,
                lip_movement=0.0
            )
        
        # Get recent MAR history
        with self.lock:
            if face.track_id not in self.mar_histories:
                return ActiveSpeakerResult(
                    track_id=face.track_id,
                    is_speaking=False,
                    probability=0.0,
                    timestamp=timestamp
                )
            
            history = list(self.mar_histories[face.track_id])
        
        # Need enough samples
        if len(history) < 5:
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=False,
                probability=0.0,
                timestamp=timestamp,
                lip_movement=0.0
            )
        
        # Get recent MAR values (last 1 second)
        cutoff = timestamp - 1.0
        recent_entries = [h for h in history if h['time'] >= cutoff]
        
        if len(recent_entries) < 3:
            recent_entries = history[-10:]
        
        recent = [h['mar'] for h in recent_entries]
        
        # Calculate metrics
        mars = np.array(recent)
        std = np.std(mars)
        rng = np.max(mars) - np.min(mars)
        
        # Calculate velocity - THIS IS THE KEY METRIC
        # Velocity = how fast is the mouth CHANGING (not just open)
        velocities = []
        for i in range(1, len(recent_entries)):
            dt = recent_entries[i]['time'] - recent_entries[i-1]['time']
            if dt > 0:
                vel = abs(recent_entries[i]['mar'] - recent_entries[i-1]['mar']) / dt
                velocities.append(vel)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        
        # Speaking requires mouth to be MOVING (velocity), not just open
        # Static open mouth: high MAR, low velocity -> NOT speaking
        # Moving mouth: varying MAR, high velocity -> SPEAKING
        #
        # Velocity threshold: ~0.1-0.2 MAR units per second is typical for speech
        # Jitter on static open mouth: ~0.01-0.05 MAR units per second
        
        min_velocity_for_speech = 0.08  # Must have some mouth movement
        
        is_speaking = (
            avg_velocity > min_velocity_for_speech and  # Mouth must be MOVING
            (std > self.std_threshold or rng > self.range_threshold)  # And have variation
        )
        
        # Calculate confidence/probability
        motion_penalty = min(1.0, head_motion / self.head_motion_threshold)
        
        # Weight velocity heavily - it's the key discriminator
        velocity_score = min(1.0, avg_velocity / 0.3)
        std_score = min(1.0, std / 0.05)
        range_score = min(1.0, rng / 0.1)
        
        base_probability = velocity_score * 0.5 + std_score * 0.25 + range_score * 0.25
        probability = base_probability * (1.0 - motion_penalty * 0.5)
        
        return ActiveSpeakerResult(
            track_id=face.track_id,
            is_speaking=is_speaking,
            probability=probability,
            timestamp=timestamp,
            lip_movement=avg_velocity  # Show velocity so you can see it
        )
    
    def detect_batch(
        self,
        faces: List[FaceDetection],
        audio: Optional[np.ndarray] = None
    ) -> List[ActiveSpeakerResult]:
        """Detect speaking status for multiple faces"""
        return [self.detect(face, audio) for face in faces]
    
    def cleanup_track(self, track_id: int):
        """Remove history for a lost track"""
        with self.lock:
            if track_id in self.mar_histories:
                del self.mar_histories[track_id]
            if track_id in self.position_histories:
                del self.position_histories[track_id]


class LightASDASD(ASDBackend):
    """
    Light-ASD deep learning backend.
    Uses audio-visual correlation for robust speaker detection.
    
    Paper: "A Light Weight Model for Active Speaker Detection" (CVPR 2023)
    - 1M parameters, runs at 30+ FPS on CPU
    - Much more robust than MAR heuristic
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = get_config()
        self.model_path = model_path
        self.model = None
        self.session = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the Light-ASD ONNX model"""
        try:
            import onnxruntime as ort
            
            if self.model_path is None:
                # Try default locations
                import os
                possible_paths = [
                    "models/light_asd.onnx",
                    os.path.expanduser("~/.smait/models/light_asd.onnx"),
                    "/opt/smait/models/light_asd.onnx"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.model_path = path
                        break
            
            if self.model_path and os.path.exists(self.model_path):
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=['CPUExecutionProvider']
                )
                print(f"[ASD] Light-ASD model loaded from {self.model_path}")
            else:
                print("[ASD] Light-ASD model not found, will use MAR fallback")
                print("[ASD] To enable Light-ASD, download the model to ~/.smait/models/light_asd.onnx")
                
        except ImportError:
            print("[ASD] onnxruntime not installed, using MAR fallback")
            print("[ASD] Install with: pip install onnxruntime")
        except Exception as e:
            print(f"[ASD] Failed to load Light-ASD model: {e}")
    
    def detect(
        self,
        face: FaceDetection,
        audio: Optional[np.ndarray] = None
    ) -> ActiveSpeakerResult:
        """Detect if the face is speaking using Light-ASD"""
        
        if self.session is None or face.mouth_roi is None:
            # Fallback to no detection
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=False,
                probability=0.0,
                timestamp=face.timestamp
            )
        
        try:
            # Preprocess mouth ROI
            # Light-ASD expects: (batch, channels, time, height, width)
            # For single frame: (1, 3, 1, 96, 96)
            mouth = face.mouth_roi.astype(np.float32) / 255.0
            mouth = np.transpose(mouth, (2, 0, 1))  # HWC -> CHW
            mouth = mouth[np.newaxis, :, np.newaxis, :, :]  # Add batch and time dims
            
            # Preprocess audio (if available)
            if audio is not None:
                # Light-ASD expects mel spectrogram
                # This is simplified - real implementation needs proper audio features
                audio_features = self._extract_audio_features(audio)
            else:
                # Use dummy audio features
                audio_features = np.zeros((1, 13, 1), dtype=np.float32)
            
            # Run inference
            inputs = {
                'video': mouth,
                'audio': audio_features
            }
            
            outputs = self.session.run(None, inputs)
            probability = float(outputs[0][0])
            
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=probability > 0.5,
                probability=probability,
                timestamp=face.timestamp,
                audio_visual_sync=probability
            )
            
        except Exception as e:
            if self.config.debug:
                print(f"[ASD] Light-ASD inference error: {e}")
            
            return ActiveSpeakerResult(
                track_id=face.track_id,
                is_speaking=False,
                probability=0.0,
                timestamp=face.timestamp
            )
    
    def _extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio for Light-ASD"""
        try:
            import librosa
            
            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=16000,
                n_mfcc=13,
                n_fft=512,
                hop_length=160
            )
            
            # Shape: (1, n_mfcc, time)
            return mfcc[np.newaxis, :, :]
            
        except ImportError:
            return np.zeros((1, 13, 1), dtype=np.float32)
    
    def detect_batch(
        self,
        faces: List[FaceDetection],
        audio: Optional[np.ndarray] = None
    ) -> List[ActiveSpeakerResult]:
        """Detect speaking status for multiple faces"""
        return [self.detect(face, audio) for face in faces]


class ActiveSpeakerDetector:
    """
    High-level Active Speaker Detection manager.
    Automatically selects backend based on config and availability.
    
    Priority: LASER > Light-ASD > MAR Heuristic
    """
    
    def __init__(self):
        self.config = get_config()
        self.backend = self._create_backend()
        
        # Track primary speaker
        self.primary_speaker_id: Optional[int] = None
        self.primary_speaker_confidence: float = 0.0
    
    def _create_backend(self) -> ASDBackend:
        """Create ASD backend with automatic fallback"""
        backend_type = self.config.vision.speaker_detection_backend
        
        # Try LASER first (recommended)
        if backend_type == SpeakerDetectionBackend.LASER:
            try:
                from smait.perception.laser_asd import LASERBackend
                backend = LASERBackend(self.config.vision.asd_model_path)
                self._backend_name = "LASER" if backend.using_model else "LASER-lite"
                self._using_fallback = False
                return backend
            except Exception as e:
                print(f"[ASD] LASER failed: {e}, trying Light-ASD")
                backend_type = SpeakerDetectionBackend.LIGHT_ASD
        
        # Try Light-ASD
        if backend_type == SpeakerDetectionBackend.LIGHT_ASD:
            try:
                backend = LightASDASD(self.config.vision.asd_model_path)
                if backend.session is not None:
                    self._backend_name = "Light-ASD"
                    self._using_fallback = False
                    return backend
                else:
                    print("[ASD] Light-ASD model not found, falling back to MAR")
            except Exception as e:
                print(f"[ASD] Light-ASD failed: {e}, falling back to MAR")
        
        # Fall back to MAR heuristic
        self._backend_name = "MAR Heuristic"
        self._using_fallback = True
        return MARHeuristicASD()
    
    def process_faces(
        self,
        faces: List[FaceDetection],
        audio: Optional[np.ndarray] = None
    ) -> Tuple[List[ActiveSpeakerResult], Optional[int]]:
        """
        Process all detected faces and determine the primary speaker.
        
        Returns:
            Tuple of (list of ASD results, primary speaker track_id)
        """
        if not faces:
            self.primary_speaker_id = None
            self.primary_speaker_confidence = 0.0
            return [], None
        
        # Detect speaking status for all faces
        results = self.backend.detect_batch(faces, audio)
        
        # Find primary speaker (highest confidence among speaking faces)
        speaking_results = [r for r in results if r.is_speaking]
        
        if speaking_results:
            best = max(speaking_results, key=lambda r: r.probability)
            self.primary_speaker_id = best.track_id
            self.primary_speaker_confidence = best.probability
        else:
            # No one speaking - keep previous or clear
            if self.primary_speaker_id is not None:
                # Check if previous speaker is still visible
                visible_ids = {r.track_id for r in results}
                if self.primary_speaker_id not in visible_ids:
                    self.primary_speaker_id = None
                    self.primary_speaker_confidence = 0.0
        
        return results, self.primary_speaker_id
    
    @property
    def using_fallback(self) -> bool:
        """Whether using MAR fallback"""
        return self._using_fallback
    
    @property
    def backend_name(self) -> str:
        """Name of the active backend"""
        return self._backend_name
