"""Configuration hierarchy using dataclasses with env/file loading."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_config_instance: Optional[Config] = None


@dataclass
class ConnectionConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    heartbeat_interval_s: float = 5.0
    reconnect_max_wait_s: float = 30.0


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels_raw: int = 4
    channels_cae: int = 1
    chunk_duration_ms: int = 32  # Silero VAD requires 512 samples = 32ms at 16kHz
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    raw_buffer_seconds: float = 30.0


@dataclass
class SeparationConfig:
    model: str = "dolphin"
    use_multichannel: bool = True
    fallback_to_cae: bool = True


@dataclass
class ASRConfig:
    model: str = "nvidia/parakeet-tdt-0.6b-v2"
    confidence_threshold: float = 0.40
    hallucination_filter: bool = True


@dataclass
class EOUConfig:
    min_silence_ms: int = 300
    confidence_threshold: float = 0.7
    hard_cutoff_ms: int = 1800
    vad_silence_ms: int = 1800
    barge_in_min_speech_ms: int = 200


@dataclass
class VisionConfig:
    max_faces: int = 5
    min_face_confidence: float = 0.6
    lip_roi_size: tuple = (88, 88)


@dataclass
class GazeConfig:
    yaw_threshold: float = 30.0
    pitch_threshold: float = 20.0


@dataclass
class EngagementConfig:
    min_gaze_duration_s: float = 2.0
    disengage_gaze_timeout_s: float = 3.0
    face_area_threshold: int = 3000
    # DOA fusion parameters
    camera_fov_deg: float = 60.0       # Horizontal FOV of Jackie's camera
    doa_weight: float = 0.0            # 0.0 = ignore DOA, 1.0 = full DOA scoring (disabled until tested)
    doa_staleness_s: float = 2.0       # DOA older than this decays toward neutral
    doa_smoothing_alpha: float = 0.3   # EMA smoothing: 0=slow response, 1=no smoothing


@dataclass
class DialogueConfig:
    local_model: str = "phi-4-mini"
    api_model: str = "gpt-4o-mini"
    api_provider: str = "openai"
    max_tokens: int = 150
    temperature: float = 0.7
    max_history_turns: int = 10
    try_local_first: bool = True
    system_prompt: str = (
        "You are Jackie, a friendly AI-powered conference robot at SJSU. "
        "Keep responses to 1-3 spoken sentences. Be warm, natural, slightly playful. "
        "You ARE a robot — own it with personality. "
        "You can see faces, detect who's speaking, and hold real conversations in real-time. "
        "If speech seems garbled: 'Sorry, it's noisy — could you say that again?' "
        "Plain spoken sentences only — no lists, markdown, or formatting."
    )


@dataclass
class TTSConfig:
    model: str = "kokoro-82m"
    sample_rate: int = 24000
    stream_by_sentence: bool = True


@dataclass
class SessionConfig:
    timeout_s: float = 30.0
    face_lost_grace_s: float = 8.0
    reacquisition_window_s: float = 20.0


@dataclass
class LogConfig:
    output_dir: str = "logs"
    save_audio: bool = True
    save_video_snapshots: bool = False


@dataclass
class Config:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    eou: EOUConfig = field(default_factory=EOUConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    debug: bool = False
    show_video: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load config from a JSON file, merging with defaults."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found, using defaults", path)
            return cls()

        with open(path) as f:
            data = json.load(f)

        config = cls()
        _merge_dict_into_dataclass(config, data)
        return config

    @classmethod
    def from_env(cls) -> Config:
        """Load config overrides from environment variables.

        Environment variables use the pattern SMAIT_<SECTION>_<FIELD>,
        e.g. SMAIT_CONNECTION_PORT=9000, SMAIT_AUDIO_VAD_THRESHOLD=0.6.
        """
        config = cls()
        prefix = "SMAIT_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section_name, field_name = parts
            section = getattr(config, section_name, None)
            if section is None:
                continue
            if hasattr(section, field_name):
                current = getattr(section, field_name)
                try:
                    converted = type(current)(value)
                    setattr(section, field_name, converted)
                except (ValueError, TypeError):
                    logger.warning("Cannot convert env %s=%s to %s",
                                   key, value, type(current).__name__)

        # Top-level boolean flags
        if os.getenv("SMAIT_DEBUG"):
            config.debug = os.getenv("SMAIT_DEBUG", "").lower() in ("1", "true", "yes")
        if os.getenv("SMAIT_SHOW_VIDEO"):
            config.show_video = os.getenv("SMAIT_SHOW_VIDEO", "").lower() in ("1", "true", "yes")

        return config


def _merge_dict_into_dataclass(obj: object, data: dict) -> None:
    """Recursively merge a dict into a dataclass instance."""
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge_dict_into_dataclass(current, value)
        else:
            try:
                if isinstance(current, tuple) and isinstance(value, list):
                    setattr(obj, key, tuple(value))
                else:
                    setattr(obj, key, type(current)(value) if not isinstance(value, type(current)) else value)
            except (ValueError, TypeError):
                logger.warning("Cannot set %s to %s", key, value)


def get_config(config_path: str | Path | None = None) -> Config:
    """Get or create the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        if config_path:
            _config_instance = Config.from_file(config_path)
        else:
            _config_instance = Config.from_env()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing)."""
    global _config_instance
    _config_instance = None
