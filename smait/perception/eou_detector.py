"""Heuristic End-of-Utterance detector (Phase 5: VAD-based rewrite)."""

from __future__ import annotations

import logging
import time
from typing import Optional

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)


class EOUDetector:
    """Heuristic End-of-Utterance detector.

    Runs on CPU (no VRAM cost).

    Usage flow:
    1. AudioPipeline detects silence > 300ms after speech
    2. Current transcript fed to EOUDetector
    3. Heuristic returns P(end_of_turn) in [0, 1]
    4. If P > 0.7 -> emit END_OF_TURN -> triggers LLM response
    5. If P < 0.7 -> wait for more speech or silence
    6. Safety: if silence > 1500ms regardless of P -> force END_OF_TURN

    Phase 5 will replace the heuristic with a VAD-based EOU model.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.eou
        self._event_bus = event_bus
        self._model = None
        self._available = False

        # State
        self._current_text = ""
        self._silence_start: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._pending_turn = False

    async def init_model(self) -> None:
        """EOU model loading deferred to Phase 5 (VAD-based rewrite).
        Phase 1: mark as unavailable; heuristic predict() path is active.
        """
        logger.info(
            "EOUDetector: using heuristic fallback "
            "(VAD-based EOU rewrite in Phase 5). "
            "LiveKit turn detector is unavailable."
        )
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def update_transcript(self, text: str, timestamp: float) -> None:
        """Update the current transcript text as ASR produces results."""
        self._current_text = text
        self._last_speech_time = timestamp
        self._silence_start = None
        self._pending_turn = True

    def on_silence(self, timestamp: float) -> None:
        """Called when silence is detected after speech."""
        if not self._pending_turn:
            return

        if self._silence_start is None:
            self._silence_start = timestamp

        silence_ms = (timestamp - self._silence_start) * 1000

        # Hard cutoff: force END_OF_TURN after extended silence
        if silence_ms >= self._config.hard_cutoff_ms:
            logger.info("EOU: hard cutoff (silence=%.0fms)", silence_ms)
            self._emit_end_of_turn(timestamp, confidence=1.0, reason="hard_cutoff")
            return

        # Check EOU model after minimum silence
        if silence_ms >= self._config.min_silence_ms:
            confidence = self.predict(self._current_text)

            if confidence >= self._config.confidence_threshold:
                logger.info("EOU: model triggered (P=%.2f, silence=%.0fms)",
                            confidence, silence_ms)
                self._emit_end_of_turn(timestamp, confidence, reason="model")
            else:
                logger.debug("EOU: waiting (P=%.2f < %.2f, silence=%.0fms)",
                             confidence, self._config.confidence_threshold, silence_ms)

    def predict(self, text: str) -> float:
        """Predict P(end_of_turn) for the given transcript text.

        Returns a float in [0, 1]. Always uses heuristic in Phase 1.
        """
        if not text.strip():
            return 0.0

        return self._heuristic_eou(text)

    def _heuristic_eou(self, text: str) -> float:
        """Heuristic EOU prediction when model is unavailable.

        Simple rules based on punctuation and sentence structure.
        """
        text = text.strip()
        if not text:
            return 0.0

        # Questions are strong end-of-turn signals
        if text.endswith("?"):
            return 0.9

        # Statements ending with period
        if text.endswith(".") or text.endswith("!"):
            return 0.8

        # Short complete-sounding phrases
        words = text.split()
        if len(words) >= 3:
            return 0.6

        # Single/double words — probably not done
        return 0.3

    def _emit_end_of_turn(self, timestamp: float, confidence: float, reason: str) -> None:
        """Emit END_OF_TURN event and reset state."""
        self._event_bus.emit(EventType.END_OF_TURN, {
            "text": self._current_text,
            "confidence": confidence,
            "reason": reason,
            "timestamp": timestamp,
            "silence_ms": (timestamp - self._silence_start) * 1000 if self._silence_start else 0,
        })
        self._pending_turn = False
        self._silence_start = None
        self._current_text = ""

    def reset(self) -> None:
        """Reset detector state (e.g., on session end)."""
        self._current_text = ""
        self._silence_start = None
        self._last_speech_time = None
        self._pending_turn = False
