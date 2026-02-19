"""
SMAIT HRI System v2.0 - Configuration
Supports: Real hardware, Isaac Sim, ROS 2, and standalone modes
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json


class DeploymentMode(Enum):
    """System deployment mode"""
    STANDALONE = "standalone"      # Direct hardware access (laptop/desktop)
    ROS2 = "ros2"                  # ROS 2 nodes (real robot)
    ISAAC_SIM = "isaac_sim"        # Isaac Sim simulation
    

class ASRBackend(Enum):
    """Speech recognition backend"""
    PARAKEET_TDT = "parakeet_tdt"      # NVIDIA NeMo, streaming, SOTA (recommended)
    FASTER_WHISPER = "faster_whisper"  # Local, chunk-based (fallback)
    AWS_TRANSCRIBE = "aws_transcribe"  # Cloud, requires internet
    WHISPER_CPP = "whisper_cpp"        # Ultra-light, for embedded


class SpeakerDetectionBackend(Enum):
    """Speaker detection method"""
    LASER = "laser"                    # Landmark-assisted (recommended)
    LIGHT_ASD = "light_asd"            # Deep learning fallback
    MAR_HEURISTIC = "mar_heuristic"    # Legacy geometric heuristic


class LLMBackend(Enum):
    """LLM provider"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


@dataclass
class AudioConfig:
    """Audio pipeline configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30      # VAD chunk size
    buffer_seconds: float = 30.0     # Ring buffer duration
    vad_threshold: float = 0.5       # Voice activity threshold
    silence_duration_ms: int = 500   # Silence before end-of-utterance
    min_speech_duration_ms: int = 250  # Minimum speech to process


@dataclass
class ASRConfig:
    """ASR configuration"""
    backend: ASRBackend = ASRBackend.PARAKEET_TDT  # Default to best option
    
    # Parakeet TDT settings (NeMo)
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"  # HuggingFace model ID
    parakeet_streaming: bool = True     # Enable streaming inference
    parakeet_chunk_seconds: float = 2.0 # Chunk size for streaming
    
    # faster-whisper settings (fallback)
    model_size: str = "base.en"      # tiny.en, base.en, small.en, medium.en
    device: str = "auto"             # auto, cpu, cuda
    compute_type: str = "auto"       # auto, int8, float16, float32
    language: str = "en"
    beam_size: int = 5
    
    # AWS Transcribe (if using cloud)
    aws_region: str = "us-west-2"
    
    # Streaming options
    enable_partial_results: bool = True
    partial_result_stability: str = "medium"  # low, medium, high


@dataclass
class VisionConfig:
    """Vision pipeline configuration"""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    
    # Face detection
    min_face_confidence: float = 0.6
    min_face_area: int = 3000
    max_faces: int = 5               # Track multiple faces
    
    # Speaker detection
    speaker_detection_backend: SpeakerDetectionBackend = SpeakerDetectionBackend.LASER
    asd_model_path: Optional[str] = None  # Path to ASD model (LASER or Light-ASD)
    asd_threshold: float = 0.5       # Active speaker probability threshold
    
    # LASER-specific settings
    laser_use_landmarks: bool = True  # Use MediaPipe landmarks (recommended)
    laser_audio_window_ms: int = 640  # Audio window for AV sync (40 frames @ 16kHz)
    laser_consistency_weight: float = 0.5  # Weight for consistency loss at inference
    
    # Legacy MAR settings (fallback)
    mar_movement_std: float = 0.02
    mar_movement_range: float = 0.05
    mar_movement_velocity: float = 0.01


@dataclass
class SessionConfig:
    """Session management configuration"""
    timeout_seconds: float = 45.0              # More generous silence timeout
    face_lost_grace_seconds: float = 10.0      # 10s grace before ending (was 5)
    reacquisition_window_seconds: float = 15.0  # Window to re-find user
    min_engagement_confidence: float = 0.3
    
    # Enrollment (legacy, will be replaced by implicit detection)
    require_explicit_enrollment: bool = False
    enrollment_phrases: int = 1


@dataclass
class DialogueConfig:
    """Dialogue/LLM configuration"""
    llm_backend: LLMBackend = LLMBackend.OPENAI
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 150
    temperature: float = 0.7
    
    # Conversation memory
    max_history_turns: int = 10      # Sliding window
    enable_summarization: bool = False
    
    # System prompt
    system_prompt: str = (
        "You are 'Jackie', a friendly service robot at SJSU (San Jose State University). "
        "You greet visitors, answer questions, and help with directions. "
        "Keep responses short (1-3 sentences) — you're speaking out loud, not typing. "
        "Be warm, natural, and slightly playful. Never say 'As an AI' or 'I don't have feelings.' "
        "If the message is garbled or unclear, just say 'Sorry, could you say that again?' "
        "IMPORTANT: Your responses will be spoken through TTS, so keep them conversational "
        "and avoid bullet points, lists, or markdown formatting."
    )
    
    # TTS (future)
    enable_tts: bool = False
    tts_backend: str = "pyttsx3"


@dataclass 
class ROS2Config:
    """ROS 2 specific configuration"""
    node_name: str = "smait_hri"
    namespace: str = ""
    
    # Topics
    camera_topic: str = "/camera/image_raw"
    audio_topic: str = "/audio/raw"
    asr_topic: str = "/smait/asr/text"
    response_topic: str = "/smait/response"
    engagement_topic: str = "/smait/engagement"
    
    # QoS
    qos_depth: int = 10


@dataclass
class IsaacSimConfig:
    """Isaac Sim specific configuration"""
    # Scene settings
    scene_path: str = ""
    
    # Human simulation
    num_simulated_humans: int = 1
    human_animation: str = "talking"  # talking, idle, walking
    
    # Synthetic audio (for testing)
    use_synthetic_audio: bool = True
    synthetic_audio_path: str = ""
    
    # Camera prim path in USD
    camera_prim: str = "/World/Robot/Camera"


@dataclass
class Config:
    """Main configuration container"""
    # Deployment
    mode: DeploymentMode = DeploymentMode.STANDALONE
    debug: bool = True
    show_video: bool = True
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    ros2: ROS2Config = field(default_factory=ROS2Config)
    isaac_sim: IsaacSimConfig = field(default_factory=IsaacSimConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()
        
        # Deployment mode
        mode_str = os.getenv("SMAIT_MODE", "standalone").lower()
        config.mode = DeploymentMode(mode_str)
        
        # Debug settings
        config.debug = os.getenv("SMAIT_DEBUG", "1") == "1"
        config.show_video = os.getenv("SMAIT_SHOW_VIDEO", "1") == "1"
        
        # Hardware
        config.vision.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        config.asr.aws_region = os.getenv("AWS_REGION", "us-west-2")
        
        # ASR backend
        asr_backend = os.getenv("SMAIT_ASR_BACKEND", "parakeet_tdt")
        try:
            config.asr.backend = ASRBackend(asr_backend)
        except ValueError:
            print(f"[CONFIG] Unknown ASR backend '{asr_backend}', falling back to faster_whisper")
            config.asr.backend = ASRBackend.FASTER_WHISPER
        config.asr.model_size = os.getenv("WHISPER_MODEL", "base.en")
        config.asr.parakeet_model = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
        
        # Speaker detection
        asd_backend = os.getenv("SMAIT_ASD_BACKEND", "laser")
        try:
            config.vision.speaker_detection_backend = SpeakerDetectionBackend(asd_backend)
        except ValueError:
            print(f"[CONFIG] Unknown ASD backend '{asd_backend}', falling back to MAR heuristic")
            config.vision.speaker_detection_backend = SpeakerDetectionBackend.MAR_HEURISTIC
        
        # LLM
        llm_backend = os.getenv("SMAIT_LLM_BACKEND", "openai")
        config.dialogue.llm_backend = LLMBackend(llm_backend)
        config.dialogue.model_name = os.getenv("SMAIT_LLM_MODEL", "gpt-4o-mini")
        
        return config
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Recursively build config from dictionary"""
        config = cls()
        
        if "mode" in data:
            config.mode = DeploymentMode(data["mode"])
        if "debug" in data:
            config.debug = data["debug"]
        if "show_video" in data:
            config.show_video = data["show_video"]
            
        # Sub-configs would need more elaborate handling
        # This is simplified for now
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary"""
        return {
            "mode": self.mode.value,
            "debug": self.debug,
            "show_video": self.show_video,
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "chunk_duration_ms": self.audio.chunk_duration_ms,
            },
            "asr": {
                "backend": self.asr.backend.value,
                "model_size": self.asr.model_size,
            },
            "vision": {
                "speaker_detection_backend": self.vision.speaker_detection_backend.value,
            },
        }
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config):
    """Set global configuration"""
    global _config
    _config = config
