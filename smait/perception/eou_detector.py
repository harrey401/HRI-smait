"""VAD-prob-based End-of-Utterance detector (Phase 05: primary path).

Architecture
------------
Primary path (Phase 05+): feed_vad_prob()
  AudioPipeline feeds per-chunk VAD speech probabilities.
  Silence is counted sample-by-sample; END_OF_TURN fires when silence
  accumulates >= vad_silence_ms (default 1800ms = 28800 samples at 16kHz).

Fallback path (heuristic): predict() / _heuristic_eou()
  Used when VAD probabilities are unavailable.
  Also reachable via on_silence() / update_transcript().

Hysteresis thresholds (Option B from Phase 05 research):
  speech_prob >= 0.50  -> enter speech (SPEECH_ENTER)
  speech_prob <  0.35  -> exit speech (SPEECH_EXIT), start accumulating silence
  0.35 <= speech_prob < 0.50 -> hysteresis zone, no state change
"""

from __future__ import annotations

import logging
from typing import Optional

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

_SPEECH_ENTER_THRESHOLD = 0.50
_SPEECH_EXIT_THRESHOLD = 0.35


class EOUDetector:
    """VAD-prob-based End-of-Utterance detector.

    Runs on CPU (no VRAM cost).

    Primary path — feed_vad_prob():
      1. AudioPipeline calls feed_vad_prob(speech_prob, n_samples, timestamp) per chunk.
      2. If speech_prob >= 0.50: enter speech state, reset silence counter.
      3. If speech_prob < 0.35 and in speech: accumulate silence samples.
         When accumulated >= vad_silence_ms threshold: emit END_OF_TURN.
      4. 0.35–0.50 hysteresis zone: no state change.

    Fallback path — on_silence() / predict():
      Used by callers that feed silence wall-clock time instead of VAD probs.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.eou
        self._event_bus = event_bus
        self._model = None
        self._available = False

        # Heuristic / fallback state
        self._current_text = ""
        self._silence_start: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._pending_turn = False

        # VAD-based EOU state
        self._in_speech: bool = False
        self._silence_sample_count: int = 0
        self._silence_threshold_samples: int = int(
            self._config.vad_silence_ms / 1000 * 16000
        )

    async def init_model(self) -> None:
        """Model loading deferred to lab phase.

        Phase 05: VAD-based EOU via feed_vad_prob() is the primary path.
        Heuristic predict() remains as fallback.
        """
        logger.info(
            "EOUDetector: VAD-based EOU active (vad_silence_ms=%d). "
            "Heuristic fallback also available.",
            self._config.vad_silence_ms,
        )
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # VAD-based primary path
    # ------------------------------------------------------------------

    def feed_vad_prob(
        self, speech_prob: float, n_samples: int, timestamp: float
    ) -> None:
        """Process one VAD chunk.

        Args:
            speech_prob: Speech probability in [0, 1] from a VAD model.
            n_samples:   Number of audio samples in this chunk (e.g. 480 for 30ms at 16kHz).
            timestamp:   Chunk end timestamp in seconds (for event metadata).
        """
        if speech_prob >= _SPEECH_ENTER_THRESHOLD:
            # --- Enter / stay in speech ---
            self._in_speech = True
            self._silence_sample_count = 0
            self._pending_turn = True

        elif speech_prob < _SPEECH_EXIT_THRESHOLD:
            # --- Exit threshold crossed ---
            if self._in_speech:
                self._silence_sample_count += n_samples
                if self._silence_sample_count >= self._silence_threshold_samples:
                    self._emit_end_of_turn(
                        timestamp, confidence=1.0, reason="vad_silence"
                    )
        # else: hysteresis zone [0.35, 0.50) — no state change

    # ------------------------------------------------------------------
    # Heuristic fallback path
    # ------------------------------------------------------------------

    def update_transcript(self, text: str, timestamp: float) -> None:
        """Update the current transcript text as ASR produces results."""
        self._current_text = text
        self._last_speech_time = timestamp
        self._silence_start = None
        self._pending_turn = True

    def on_silence(self, timestamp: float) -> None:
        """Called when silence is detected after speech (wall-clock path)."""
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
                logger.info(
                    "EOU: model triggered (P=%.2f, silence=%.0fms)",
                    confidence,
                    silence_ms,
                )
                self._emit_end_of_turn(timestamp, confidence, reason="model")
            else:
                logger.debug(
                    "EOU: waiting (P=%.2f < %.2f, silence=%.0fms)",
                    confidence,
                    self._config.confidence_threshold,
                    silence_ms,
                )

    def predict(self, text: str) -> float:
        """Predict P(end_of_turn) for the given transcript text.

        Returns a float in [0, 1]. Always uses heuristic (no model loaded).
        """
        if not text.strip():
            return 0.0

        return self._heuristic_eou(text)

    def _heuristic_eou(self, text: str) -> float:
        """Heuristic EOU prediction.

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

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _emit_end_of_turn(
        self, timestamp: float, confidence: float, reason: str
    ) -> None:
        """Emit END_OF_TURN event and reset shared + VAD state."""
        silence_ms: float = 0.0
        if self._silence_start is not None:
            silence_ms = (timestamp - self._silence_start) * 1000

        self._event_bus.emit(
            EventType.END_OF_TURN,
            {
                "text": self._current_text,
                "confidence": confidence,
                "reason": reason,
                "timestamp": timestamp,
                "silence_ms": silence_ms,
            },
        )

        # Reset heuristic state
        self._pending_turn = False
        self._silence_start = None
        self._current_text = ""

        # Reset VAD state
        self._in_speech = False
        self._silence_sample_count = 0

    def reset(self) -> None:
        """Reset all detector state (e.g., on session end)."""
        # Heuristic state
        self._current_text = ""
        self._silence_start = None
        self._last_speech_time = None
        self._pending_turn = False

        # VAD state
        self._in_speech = False
        self._silence_sample_count = 0
