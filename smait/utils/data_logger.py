"""
SMAIT HRI System v2.0 - Data Logger for March 3 Demo
Structured JSON logging per interaction session + raw audio saving.

Usage:
    from smait.utils.data_logger import DataLogger

    logger = DataLogger(output_dir="logs/march3")
    logger.start_session(engagement_info)
    logger.log_turn(turn_data)
    logger.end_session(reason)

Each session produces:
- session_{id}.json — structured interaction data
- audio/session_{id}_turn_{n}.wav — raw audio segments (for ASR debugging/WER)
"""

import json
import os
import time
import wave
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class EngagementInfo:
    """Engagement trigger data"""
    proximity_m: float = 0.0
    head_yaw_deg: float = 0.0
    greeting_type: str = ""  # proactive, reactive, re-engagement
    face_area: int = 0
    attention_ok: bool = False
    proximity_ok: bool = False


@dataclass
class CAEStatus:
    """Jackie CAE SDK hardware filter status"""
    aec: bool = False
    beamforming: bool = False
    noise_suppression: bool = False
    timestamp: float = 0.0


@dataclass
class TurnData:
    """Per-turn interaction data"""
    timestamp: float = 0.0
    turn_index: int = 0

    # ASR
    user_text: str = ""
    asr_confidence: float = 0.0
    asr_latency_ms: float = 0.0
    segment_duration_ms: float = 0.0

    # ASD / Verification
    asd_score: float = 0.0
    verification_result: str = ""  # ACCEPT, REJECT, NO_FACE
    verification_reason: str = ""
    concurrent_faces: int = 0

    # Dialogue
    robot_text: str = ""
    llm_latency_ms: float = 0.0
    llm_model: str = ""

    # TTS
    tts_latency_ms: float = 0.0

    # Computed
    total_response_time_ms: float = 0.0

    # Quality flags
    error_recovery: bool = False  # robot asked user to repeat
    phantom_transcript: bool = False  # filtered hallucination detected
    wrong_speaker_rejection: bool = False


@dataclass
class SessionLog:
    """Complete session log"""
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0

    # Engagement
    engagement: Optional[Dict] = None

    # Hardware status
    cae_status: Optional[Dict] = None
    doa_angles: List[float] = field(default_factory=list)

    # Turns
    turns: List[Dict] = field(default_factory=list)
    total_turns: int = 0

    # Session outcome
    end_reason: str = ""  # farewell, timeout, face_lost

    # HRI Success Checklist (auto-scored)
    checklist: Dict[str, bool] = field(default_factory=dict)
    interaction_score: int = 0

    # Environment
    ambient_noise_db: float = 0.0


class DataLogger:
    """
    Structured data logger for SMAIT HRI interactions.
    
    Logs per-session JSON files and optionally saves raw audio
    segments for post-hoc ASR analysis.
    """

    def __init__(
        self,
        output_dir: str = "logs/march3",
        save_audio: bool = True,
        save_video_snapshots: bool = False,
        sample_rate: int = 16000
    ):
        self.output_dir = Path(output_dir)
        self.save_audio = save_audio
        self.save_video_snapshots = save_video_snapshots
        self.sample_rate = sample_rate

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_audio:
            (self.output_dir / "audio").mkdir(exist_ok=True)

        # Current session state
        self._session: Optional[SessionLog] = None
        self._session_id: str = ""
        self._turn_count: int = 0
        self._phantom_count: int = 0
        self._wrong_speaker_count: int = 0
        self._had_proactive_greeting: bool = False
        self._first_utterance_accepted: bool = False
        self._first_utterance_done: bool = False

        # CAE status (updated from WebSocket)
        self._latest_cae: Optional[CAEStatus] = None

        # DOA buffer
        self._doa_buffer: List[float] = []

        # Noise tracking
        self._noise_samples: List[float] = []

        print(f"[DATA-LOG] Initialized → {self.output_dir}")

    # ── Session lifecycle ────────────────────────────────────────────────

    def start_session(
        self,
        engagement: Optional[EngagementInfo] = None,
        proactive_greeting: bool = False
    ):
        """Start logging a new interaction session"""
        self._session_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self._turn_count = 0
        self._phantom_count = 0
        self._wrong_speaker_count = 0
        self._had_proactive_greeting = proactive_greeting
        self._first_utterance_accepted = False
        self._first_utterance_done = False
        self._doa_buffer = []
        self._noise_samples = []

        self._session = SessionLog(
            session_id=self._session_id,
            start_time=time.time(),
            engagement=asdict(engagement) if engagement else None,
            cae_status=asdict(self._latest_cae) if self._latest_cae else None,
        )

        print(f"[DATA-LOG] Session started: {self._session_id}")

    def end_session(self, reason: str = "unknown"):
        """End current session, compute scores, write to disk"""
        if self._session is None:
            return

        self._session.end_time = time.time()
        self._session.duration_s = self._session.end_time - self._session.start_time
        self._session.total_turns = self._turn_count
        self._session.end_reason = reason
        self._session.doa_angles = self._doa_buffer.copy()

        # Ambient noise (mean RMS in dB)
        if self._noise_samples:
            mean_rms = sum(self._noise_samples) / len(self._noise_samples)
            self._session.ambient_noise_db = 20 * np.log10(max(mean_rms, 1e-10))

        # Auto-score HRI checklist
        self._compute_checklist()

        # Write JSON
        self._write_session()

        session_id = self._session_id
        score = self._session.interaction_score
        print(f"[DATA-LOG] Session ended: {session_id} | score={score}/7 | reason={reason} | turns={self._turn_count}")

        self._session = None
        self._session_id = ""

    # ── Per-turn logging ─────────────────────────────────────────────────

    def log_turn(self, turn: TurnData):
        """Log a complete interaction turn (user spoke → robot responded)"""
        if self._session is None:
            return

        self._turn_count += 1
        turn.turn_index = self._turn_count
        turn.timestamp = time.time()

        # Track first utterance
        if not self._first_utterance_done:
            self._first_utterance_done = True
            if turn.verification_result == "ACCEPT" and not turn.error_recovery:
                self._first_utterance_accepted = True

        self._session.turns.append(asdict(turn))

    def log_rejected_transcript(
        self,
        text: str,
        reason: str,
        asd_score: float = 0.0,
        is_phantom: bool = False,
        is_wrong_speaker: bool = False
    ):
        """Log a rejected/filtered transcript (not a full turn, but important data)"""
        if self._session is None:
            return

        if is_phantom:
            self._phantom_count += 1
        if is_wrong_speaker:
            self._wrong_speaker_count += 1

        # Store as a lightweight entry in turns
        self._session.turns.append({
            "type": "rejected",
            "timestamp": time.time(),
            "text": text,
            "reason": reason,
            "asd_score": asd_score,
            "is_phantom": is_phantom,
            "is_wrong_speaker": is_wrong_speaker,
        })

    # ── Audio saving ─────────────────────────────────────────────────────

    def save_audio_segment(
        self,
        audio: np.ndarray,
        turn_index: Optional[int] = None,
        label: str = ""
    ) -> Optional[str]:
        """
        Save raw audio segment as WAV for post-hoc analysis.
        Returns the file path, or None if saving is disabled.
        """
        if not self.save_audio or self._session is None:
            return None

        if turn_index is None:
            turn_index = self._turn_count + 1

        suffix = f"_{label}" if label else ""
        filename = f"session_{self._session_id}_turn_{turn_index}{suffix}.wav"
        filepath = self.output_dir / "audio" / filename

        try:
            # Ensure int16
            if audio.dtype != np.int16:
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio.tobytes())

            return str(filepath)

        except Exception as e:
            print(f"[DATA-LOG] Failed to save audio: {e}")
            return None

    # ── Hardware status updates ──────────────────────────────────────────

    def update_cae_status(self, aec: bool, beamforming: bool, noise_suppression: bool):
        """Update CAE SDK status (called from WebSocket handler)"""
        self._latest_cae = CAEStatus(
            aec=aec,
            beamforming=beamforming,
            noise_suppression=noise_suppression,
            timestamp=time.time()
        )

        # Warn if critical features are off
        if not beamforming:
            print("[DATA-LOG] ⚠️  WARNING: Jackie beamforming is OFF!")
        if not noise_suppression:
            print("[DATA-LOG] ⚠️  WARNING: Jackie noise suppression is OFF!")
        if not aec:
            print("[DATA-LOG] ⚠️  WARNING: Jackie AEC is OFF!")

        # Update current session if active
        if self._session:
            self._session.cae_status = asdict(self._latest_cae)

    def log_doa_angle(self, angle: float):
        """Log Direction of Arrival angle"""
        self._doa_buffer.append(angle)

    def log_noise_level(self, rms: float):
        """Log ambient noise RMS sample"""
        self._noise_samples.append(rms)

    # ── Scoring ──────────────────────────────────────────────────────────

    def _compute_checklist(self):
        """Auto-compute the HRI success checklist"""
        if self._session is None:
            return

        s = self._session
        checklist = {}

        # 1. Engagement detected correctly
        checklist["engagement_detected"] = s.engagement is not None

        # 2. Proactive greeting delivered
        checklist["proactive_greeting"] = self._had_proactive_greeting

        # 3. First utterance transcribed + accepted without repeat
        checklist["first_utterance_clean"] = self._first_utterance_accepted

        # 4. Multi-turn conversation (≥3 user turns)
        accepted_turns = sum(
            1 for t in s.turns
            if isinstance(t, dict) and t.get("type") != "rejected"
        )
        checklist["multi_turn"] = accepted_turns >= 3

        # 5. No phantom transcripts accepted during session
        # (phantoms that got through filtering into accepted turns)
        phantoms_accepted = sum(
            1 for t in s.turns
            if isinstance(t, dict)
            and t.get("type") != "rejected"
            and t.get("phantom_transcript", False)
        )
        checklist["no_phantoms"] = phantoms_accepted == 0

        # 6. No wrong-speaker rejections during active session
        checklist["no_wrong_speaker"] = self._wrong_speaker_count == 0

        # 7. Clean session end (farewell)
        checklist["clean_farewell"] = s.end_reason == "farewell"

        s.checklist = checklist
        s.interaction_score = sum(1 for v in checklist.values() if v)

    # ── File I/O ─────────────────────────────────────────────────────────

    def _write_session(self):
        """Write session log to JSON file"""
        if self._session is None:
            return

        filename = f"session_{self._session_id}.json"
        filepath = self.output_dir / filename

        try:
            data = asdict(self._session)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[DATA-LOG] Failed to write session: {e}")

    # ── Aggregate analysis (run post-event) ──────────────────────────────

    @staticmethod
    def analyze_event(log_dir: str) -> Dict[str, Any]:
        """
        Analyze all session logs from an event.
        Run this after March 2 to get aggregate metrics.
        
        Usage:
            stats = DataLogger.analyze_event("logs/march3")
            print(json.dumps(stats, indent=2))
        """
        log_path = Path(log_dir)
        sessions = []

        for f in sorted(log_path.glob("session_*.json")):
            with open(f) as fh:
                sessions.append(json.load(fh))

        if not sessions:
            return {"error": "No sessions found"}

        total = len(sessions)
        durations = [s["duration_s"] for s in sessions]
        turn_counts = [s["total_turns"] for s in sessions]
        scores = [s["interaction_score"] for s in sessions]

        # End reasons
        end_reasons = {}
        for s in sessions:
            r = s.get("end_reason", "unknown")
            end_reasons[r] = end_reasons.get(r, 0) + 1

        # Checklist pass rates
        checklist_rates = {}
        for s in sessions:
            for key, val in s.get("checklist", {}).items():
                if key not in checklist_rates:
                    checklist_rates[key] = {"pass": 0, "total": 0}
                checklist_rates[key]["total"] += 1
                if val:
                    checklist_rates[key]["pass"] += 1

        for key in checklist_rates:
            r = checklist_rates[key]
            r["rate"] = round(r["pass"] / max(r["total"], 1), 3)

        # Response latencies
        all_response_times = []
        all_asr_latencies = []
        error_recovery_count = 0
        error_recovery_success = 0

        for s in sessions:
            for t in s.get("turns", []):
                if t.get("type") == "rejected":
                    continue
                if t.get("total_response_time_ms", 0) > 0:
                    all_response_times.append(t["total_response_time_ms"])
                if t.get("asr_latency_ms", 0) > 0:
                    all_asr_latencies.append(t["asr_latency_ms"])
                if t.get("error_recovery"):
                    error_recovery_count += 1

        # CAE status consistency
        cae_always_on = all(
            s.get("cae_status", {}).get("beamforming", False)
            and s.get("cae_status", {}).get("noise_suppression", False)
            for s in sessions if s.get("cae_status")
        )

        return {
            "total_sessions": total,
            "mean_duration_s": round(sum(durations) / total, 1),
            "mean_turns": round(sum(turn_counts) / total, 1),
            "mean_interaction_score": round(sum(scores) / total, 2),
            "success_rate_6_of_7": round(
                sum(1 for s in scores if s >= 6) / total, 3
            ),
            "success_rate_7_of_7": round(
                sum(1 for s in scores if s >= 7) / total, 3
            ),
            "end_reasons": end_reasons,
            "checklist_pass_rates": checklist_rates,
            "mean_response_time_ms": round(
                sum(all_response_times) / max(len(all_response_times), 1), 1
            ),
            "mean_asr_latency_ms": round(
                sum(all_asr_latencies) / max(len(all_asr_latencies), 1), 1
            ),
            "error_recovery_events": error_recovery_count,
            "cae_hardware_always_on": cae_always_on,
        }


# ── Global instance ──────────────────────────────────────────────────────

_logger: Optional[DataLogger] = None


def get_data_logger() -> Optional[DataLogger]:
    """Get the global data logger instance"""
    return _logger


def init_data_logger(output_dir: str = "logs/march3", **kwargs) -> DataLogger:
    """Initialize and return the global data logger"""
    global _logger
    _logger = DataLogger(output_dir=output_dir, **kwargs)
    return _logger
