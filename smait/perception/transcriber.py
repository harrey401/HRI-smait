"""ASR orchestration: separated audio -> transcript + filters."""

from __future__ import annotations

import logging
from typing import Optional

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.asr import ParakeetASR, TranscriptResult
from smait.perception.dolphin_separator import SeparationResult

logger = logging.getLogger(__name__)

# Known ASR hallucination phrases (Issue #6 mitigation)
HALLUCINATION_PHRASES = frozenset({
    "yeah", "okay", "thank you", "bye", "thanks for watching",
    "thanks", "yes", "no", "hmm", "uh", "um", "oh",
    "thank you for watching", "please subscribe",
    "see you next time", "goodbye",
})

# Confidence threshold for hallucination filter
HALLUCINATION_CONFIDENCE_THRESHOLD = 0.60


class Transcriber:
    """Orchestrates the ASR pipeline with filtering.

    1. Receive SPEECH_SEPARATED event (clean audio from Dolphin)
    2. Run Parakeet ASR -> TranscriptResult
    3. Apply confidence filter:
       - Short utterance (<8 words) + confidence < 0.40 -> REJECT
       - Known hallucination phrases at low confidence -> REJECT
    4. If passes: emit TRANSCRIPT_READY
    5. If rejected: emit TRANSCRIPT_REJECTED with reason
    """

    def __init__(self, config: Config, event_bus: EventBus, asr: ParakeetASR) -> None:
        self._config = config.asr
        self._event_bus = event_bus
        self._asr = asr

    async def process_separated_audio(self, separation: SeparationResult, start_time: float = 0.0, end_time: float = 0.0) -> Optional[TranscriptResult]:
        """Run ASR on separated audio and apply filters.

        Returns the TranscriptResult if it passes filters, None if rejected.
        """
        result = self._asr.transcribe(
            separation.separated_audio,
            start_time=start_time,
            end_time=end_time,
        )

        if result is None:
            self._event_bus.emit(EventType.TRANSCRIPT_REJECTED, {
                "reason": "asr_unavailable",
            })
            return None

        # Empty transcript
        if not result.text.strip():
            self._event_bus.emit(EventType.TRANSCRIPT_REJECTED, {
                "reason": "empty_transcript",
            })
            return None

        # Apply filters
        rejection_reason = self._check_filters(result)

        if rejection_reason:
            logger.info("Transcript rejected: '%s' (reason: %s, conf=%.2f)",
                        result.text, rejection_reason, result.confidence)
            self._event_bus.emit(EventType.TRANSCRIPT_REJECTED, {
                "text": result.text,
                "confidence": result.confidence,
                "reason": rejection_reason,
            })
            return None

        # Passed all filters
        logger.info("Transcript accepted: '%s' (conf=%.2f)", result.text, result.confidence)
        self._event_bus.emit(EventType.TRANSCRIPT_READY, result)
        return result

    def _check_filters(self, result: TranscriptResult) -> Optional[str]:
        """Apply confidence and hallucination filters.

        Returns rejection reason string, or None if passes all filters.
        """
        text = result.text.strip()
        text_lower = text.lower().rstrip(".!?,")
        word_count = len(text.split())

        # Short utterance + low confidence -> reject (Issue #6)
        if word_count < 8 and result.confidence < self._config.confidence_threshold:
            return f"low_confidence_short ({word_count} words, conf={result.confidence:.2f})"

        # Known hallucination phrases at moderate-low confidence
        if self._config.hallucination_filter:
            if text_lower in HALLUCINATION_PHRASES:
                if result.confidence < HALLUCINATION_CONFIDENCE_THRESHOLD:
                    return f"hallucination_phrase ('{text_lower}', conf={result.confidence:.2f})"

        return None
