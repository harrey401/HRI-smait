"""Structured JSON interaction logger + HRI checklist scoring."""

from __future__ import annotations

import json
import logging
import os
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from smait.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TurnLog:
    """Per-turn metrics."""
    turn_number: int = 0
    user_text: str = ""
    asr_confidence: float = 0.0
    asr_latency_ms: float = 0.0
    separation_snr: float = 0.0
    dolphin_confidence: float = 0.0
    doa_angle: int = -1
    doa_face_alignment_deg: float = 0.0
    eou_confidence: float = 0.0
    silence_before_turn_ms: float = 0.0
    robot_text: str = ""
    llm_latency_ms: float = 0.0
    llm_model_used: str = ""
    tts_latency_ms: float = 0.0
    total_response_time_ms: float = 0.0
    verification_result: str = ""
    verification_reason: str = ""


@dataclass
class HRIChecklist:
    """HRI success checklist (auto-scored 0-7)."""
    engagement_detected: bool = False
    proactive_greeting: bool = False
    first_utterance_clean: bool = False
    multi_turn_3plus: bool = False
    no_phantoms: bool = False
    no_wrong_speaker: bool = False
    clean_farewell: bool = False

    @property
    def score(self) -> int:
        return sum([
            self.engagement_detected,
            self.proactive_greeting,
            self.first_utterance_clean,
            self.multi_turn_3plus,
            self.no_phantoms,
            self.no_wrong_speaker,
            self.clean_farewell,
        ])


@dataclass
class SessionLog:
    """Complete session log."""
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    engagement_info: dict = field(default_factory=dict)
    cae_status: dict = field(default_factory=dict)
    doa_angles: list[int] = field(default_factory=list)
    turns: list[TurnLog] = field(default_factory=list)
    hri_checklist: HRIChecklist = field(default_factory=HRIChecklist)
    errors: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return self.hri_checklist.score


class DataLogger:
    """Structured JSON per session.

    Output: logs/<event-name>/<session-id>.json
    Optional: save WAV files (raw, separated, response audio)
    """

    def __init__(self, config: Config, event_name: str = "default") -> None:
        self._config = config.logging
        self._event_name = event_name
        self._current_session: Optional[SessionLog] = None
        self._current_turn: Optional[TurnLog] = None
        self._turn_count = 0
        self._phantom_count = 0
        self._wrong_speaker_count = 0

        # Ensure output directory exists
        self._output_dir = Path(self._config.output_dir) / event_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def start_session(self, session_id: str, engagement_info: Optional[dict] = None) -> None:
        """Start logging a new session."""
        self._current_session = SessionLog(
            session_id=session_id,
            start_time=time.time(),
            engagement_info=engagement_info or {},
        )
        self._turn_count = 0
        self._phantom_count = 0
        self._wrong_speaker_count = 0
        logger.info("DataLogger: session %s started", session_id)

    def set_cae_status(self, status: dict) -> None:
        """Record CAE status at session start."""
        if self._current_session:
            self._current_session.cae_status = status
            # Warn if critical features are OFF
            if not status.get("beamforming"):
                logger.warning("CAE beamforming is OFF — audio quality may be degraded")
            if not status.get("aec"):
                logger.warning("CAE AEC is OFF — echo cancellation unavailable")

    def add_doa_angle(self, angle: int) -> None:
        """Record a DOA angle reading."""
        if self._current_session:
            self._current_session.doa_angles.append(angle)

    def start_turn(self) -> TurnLog:
        """Start a new conversation turn."""
        self._turn_count += 1
        self._current_turn = TurnLog(turn_number=self._turn_count)
        return self._current_turn

    def end_turn(self) -> None:
        """Finalize the current turn and add to session."""
        if self._current_session and self._current_turn:
            self._current_session.turns.append(self._current_turn)
            self._current_turn = None

    def record_phantom(self) -> None:
        """Record a phantom (hallucinated) transcription."""
        self._phantom_count += 1

    def record_wrong_speaker(self) -> None:
        """Record a wrong speaker attribution."""
        self._wrong_speaker_count += 1

    def end_session(self, clean_farewell: bool = False) -> Optional[SessionLog]:
        """End the session, compute HRI checklist, and save to disk."""
        if self._current_session is None:
            return None

        session = self._current_session
        session.end_time = time.time()
        session.duration = session.end_time - session.start_time

        # Auto-score HRI checklist
        session.hri_checklist.engagement_detected = len(session.engagement_info) > 0
        session.hri_checklist.proactive_greeting = self._turn_count > 0
        session.hri_checklist.first_utterance_clean = (
            len(session.turns) > 0 and
            session.turns[0].asr_confidence >= 0.40
        )
        session.hri_checklist.multi_turn_3plus = self._turn_count >= 3
        session.hri_checklist.no_phantoms = self._phantom_count == 0
        session.hri_checklist.no_wrong_speaker = self._wrong_speaker_count == 0
        session.hri_checklist.clean_farewell = clean_farewell

        # Save to disk
        self._save_session(session)
        self._current_session = None

        logger.info("DataLogger: session %s ended (score=%d/7, duration=%.1fs)",
                     session.session_id, session.score, session.duration)

        return session

    def save_audio_wav(
        self,
        audio: np.ndarray,
        filename: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Optional[str]:
        """Save audio as a WAV file for debugging.

        Returns the file path, or None if saving is disabled.
        """
        if not self._config.save_audio:
            return None

        if self._current_session is None:
            return None

        session_dir = self._output_dir / self._current_session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        filepath = session_dir / filename

        try:
            if audio.dtype == np.float32:
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)

            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            logger.debug("Saved audio: %s (%d samples)", filepath, len(audio))
            return str(filepath)

        except Exception:
            logger.exception("Failed to save audio: %s", filepath)
            return None

    def log_error(self, error: str) -> None:
        """Record an error in the session log."""
        if self._current_session:
            self._current_session.errors.append(f"{time.time():.3f}: {error}")

    def _save_session(self, session: SessionLog) -> None:
        """Write session log to JSON file."""
        filepath = self._output_dir / f"{session.session_id}.json"
        try:
            data = {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": session.duration,
                "engagement_info": session.engagement_info,
                "cae_status": session.cae_status,
                "doa_angles": session.doa_angles,
                "turns": [asdict(t) for t in session.turns],
                "hri_checklist": asdict(session.hri_checklist),
                "score": session.score,
                "errors": session.errors,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("Session log saved: %s", filepath)

        except Exception:
            logger.exception("Failed to save session log: %s", filepath)
