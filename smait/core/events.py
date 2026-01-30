"""
SMAIT HRI System v2.0 - Data Types and Events
Common data structures used across the system
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import time


# ============================================================================
# Audio Events
# ============================================================================

@dataclass
class AudioChunk:
    """Raw audio data chunk"""
    data: np.ndarray              # PCM samples (int16 or float32)
    timestamp: float              # Wall-clock time (time.time())
    sample_rate: int = 16000
    is_speech: bool = False       # VAD result
    speech_probability: float = 0.0


@dataclass
class SpeechSegment:
    """Detected speech segment with audio"""
    audio: np.ndarray             # Full audio of the segment
    start_time: float             # Wall-clock start
    end_time: float               # Wall-clock end
    duration: float = 0.0         # Duration in seconds
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class TranscriptResult:
    """ASR transcription result"""
    text: str
    is_final: bool                # True if final, False if partial
    confidence: float = 1.0
    start_time: float = 0.0       # Relative to segment start
    end_time: float = 0.0
    timestamp: float = 0.0        # Wall-clock when received
    
    # Word-level info (if available)
    words: List[Dict[str, Any]] = field(default_factory=list)
    
    # Source segment reference
    segment_id: Optional[str] = None


# ============================================================================
# Vision Events
# ============================================================================

@dataclass
class BoundingBox:
    """Face/object bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class FaceDetection:
    """Single face detection result"""
    track_id: int                 # Persistent tracking ID
    bbox: BoundingBox
    confidence: float
    timestamp: float
    
    # Landmarks (if available)
    landmarks: Optional[np.ndarray] = None  # Shape: (N, 2) or (N, 3)
    
    # Derived metrics
    mouth_roi: Optional[np.ndarray] = None  # Cropped mouth region for ASD
    
    # Legacy MAR (for fallback)
    mar: float = 0.0              # Mouth aspect ratio
    
    @property
    def center(self) -> Tuple[int, int]:
        return self.bbox.center


@dataclass
class ActiveSpeakerResult:
    """Active Speaker Detection result for a face"""
    track_id: int
    is_speaking: bool
    probability: float            # 0.0 to 1.0
    timestamp: float
    
    # Confidence breakdown (for debugging)
    audio_visual_sync: float = 0.0
    lip_movement: float = 0.0


@dataclass
class FrameResult:
    """Complete vision processing result for one frame"""
    frame: Optional[np.ndarray]   # Annotated frame (if show_video)
    faces: List[FaceDetection]
    active_speakers: List[ActiveSpeakerResult]
    timestamp: float
    
    # Primary interaction target (highest confidence active speaker)
    primary_face_id: Optional[int] = None
    
    def get_primary_face(self) -> Optional[FaceDetection]:
        """Get the face of the primary interaction target"""
        if self.primary_face_id is None:
            return None
        for face in self.faces:
            if face.track_id == self.primary_face_id:
                return face
        return None


# ============================================================================
# Session / Verification Events
# ============================================================================

class VerifyResult(Enum):
    """Speech verification outcome"""
    ACCEPT = "accept"             # Speech accepted, from enrolled user
    REJECT = "reject"             # Speech rejected (not from user)
    NO_FACE = "no_face"           # No face visible
    WARMUP = "warmup"             # System warming up / enrolling
    UNCERTAIN = "uncertain"       # Low confidence, needs retry


class SessionState(Enum):
    """Session lifecycle state"""
    IDLE = auto()                 # No active session
    DETECTING = auto()            # Looking for active speaker
    ENGAGED = auto()              # Active conversation
    PAUSED = auto()               # User temporarily away
    ENDING = auto()               # Session ending


@dataclass
class VerifyOutput:
    """Complete verification result"""
    result: VerifyResult
    text: str                     # Transcript (if accepted)
    confidence: float
    reason: str
    
    # Detailed breakdown
    face_id: Optional[int] = None
    asd_score: float = 0.0
    lip_sync_score: float = 0.0


@dataclass
class SessionInfo:
    """Current session information"""
    state: SessionState
    user_face_id: Optional[int]   # Tracked face ID of current user
    start_time: Optional[float]
    last_activity: Optional[float]
    turn_count: int = 0
    
    # Engagement metrics
    engagement_score: float = 0.0
    attention_score: float = 0.0  # Gaze/pose toward robot
    
    def elapsed_since_activity(self) -> float:
        """Seconds since last activity"""
        if self.last_activity is None:
            return float('inf')
        return time.time() - self.last_activity


# ============================================================================
# Dialogue Events
# ============================================================================

@dataclass
class DialogueTurn:
    """Single dialogue turn"""
    role: str                     # "user" or "assistant"
    content: str
    timestamp: float
    
    # Metadata
    confidence: float = 1.0
    latency_ms: float = 0.0       # Time to generate (for assistant)


@dataclass
class DialogueResponse:
    """LLM response with metadata"""
    text: str
    latency_ms: float
    token_count: int = 0
    model: str = ""
    
    # For TTS
    audio: Optional[np.ndarray] = None


# ============================================================================
# System Events (for event bus / behavior tree)
# ============================================================================

class EventType(Enum):
    """System event types for pub/sub"""
    # Audio events
    AUDIO_CHUNK = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()
    TRANSCRIPT_PARTIAL = auto()
    TRANSCRIPT_FINAL = auto()
    
    # Vision events
    FRAME_PROCESSED = auto()
    FACE_DETECTED = auto()
    FACE_LOST = auto()
    ACTIVE_SPEAKER_DETECTED = auto()
    
    # Session events
    SESSION_START = auto()
    SESSION_END = auto()
    SESSION_TIMEOUT = auto()
    USER_ENGAGED = auto()
    USER_DISENGAGED = auto()
    
    # Dialogue events
    USER_TURN = auto()
    ASSISTANT_TURN = auto()
    
    # System events
    SYSTEM_READY = auto()
    SYSTEM_ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class Event:
    """Generic event wrapper"""
    type: EventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: str = ""


# ============================================================================
# Utility Types
# ============================================================================

@dataclass
class PerformanceMetrics:
    """System performance tracking"""
    # Latencies (ms)
    audio_capture_latency: float = 0.0
    asr_latency: float = 0.0
    vision_latency: float = 0.0
    asd_latency: float = 0.0
    llm_latency: float = 0.0
    total_response_latency: float = 0.0
    
    # Rates
    audio_fps: float = 0.0
    video_fps: float = 0.0
    
    # Accuracy (from evaluation)
    asr_wer: float = 0.0
    asd_accuracy: float = 0.0
    false_accept_rate: float = 0.0
    false_reject_rate: float = 0.0
