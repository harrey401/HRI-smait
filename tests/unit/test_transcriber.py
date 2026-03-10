"""Tests for Transcriber hallucination filtering and ParakeetASR confidence extraction (Plan 05-01).

Coverage:
- Hallucination phrase rejection/acceptance based on confidence
- Short utterance low-confidence rejection
- Long utterance acceptance even at low confidence
- Empty transcript rejection
- ASR unavailable rejection
- NeMo Hypothesis word_confidence extraction path
- NeMo fallback (0.65) when word_confidence unavailable
- Expanded blocklist membership
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.asr import ParakeetASR, TranscriptResult
from smait.perception.transcriber import (
    HALLUCINATION_PHRASES,
    Transcriber,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_asr():
    """ParakeetASR with mocked transcribe() — controls returned TranscriptResult."""
    asr = MagicMock(spec=ParakeetASR)
    asr.available = True
    return asr


@pytest.fixture
def transcriber(config, event_bus, mock_asr):
    """Transcriber instance backed by a mock ASR."""
    return Transcriber(config, event_bus, mock_asr)


def _make_result(text: str, confidence: float) -> TranscriptResult:
    return TranscriptResult(
        text=text,
        confidence=confidence,
        word_timestamps=[],
        latency_ms=10.0,
    )


def _separation():
    """Minimal SeparationResult stub with separated_audio attribute."""
    sep = MagicMock()
    sep.separated_audio = MagicMock()
    return sep


def _collect_events(event_bus, event_type):
    events = []
    event_bus.subscribe(event_type, lambda data: events.append(data))
    return events


# ---------------------------------------------------------------------------
# Hallucination phrase tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hallucination_phrase_rejected(transcriber, mock_asr, event_bus):
    """'thanks for watching' at conf=0.30 is rejected (hallucination_phrase)."""
    mock_asr.transcribe.return_value = _make_result("thanks for watching", 0.30)
    rejected = _collect_events(event_bus, EventType.TRANSCRIPT_REJECTED)

    result = await transcriber.process_separated_audio(_separation())

    assert result is None
    assert len(rejected) == 1
    assert "hallucination_phrase" in rejected[0]["reason"]


@pytest.mark.asyncio
async def test_hallucination_phrase_accepted_high_conf(transcriber, mock_asr, event_bus):
    """'thank you' at conf=0.70 is accepted (above HALLUCINATION_CONFIDENCE_THRESHOLD 0.60)."""
    mock_asr.transcribe.return_value = _make_result("thank you", 0.70)
    accepted = _collect_events(event_bus, EventType.TRANSCRIPT_READY)
    rejected = _collect_events(event_bus, EventType.TRANSCRIPT_REJECTED)

    result = await transcriber.process_separated_audio(_separation())

    assert result is not None
    assert len(accepted) == 1
    assert len(rejected) == 0


# ---------------------------------------------------------------------------
# Short utterance confidence tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_short_low_conf_rejected(transcriber, mock_asr, event_bus):
    """Short utterance (3 words) at conf=0.20 is rejected (low_confidence_short)."""
    mock_asr.transcribe.return_value = _make_result("yes it is", 0.20)
    rejected = _collect_events(event_bus, EventType.TRANSCRIPT_REJECTED)

    result = await transcriber.process_separated_audio(_separation())

    assert result is None
    assert len(rejected) == 1
    assert "low_confidence_short" in rejected[0]["reason"]


@pytest.mark.asyncio
async def test_short_above_conf_accepted(transcriber, mock_asr, event_bus):
    """Short utterance (3 words) at conf=0.50 is accepted (above ASR threshold 0.40)."""
    mock_asr.transcribe.return_value = _make_result("yes it is", 0.50)
    accepted = _collect_events(event_bus, EventType.TRANSCRIPT_READY)

    result = await transcriber.process_separated_audio(_separation())

    assert result is not None
    assert len(accepted) == 1


@pytest.mark.asyncio
async def test_long_utterance_low_conf_accepted(transcriber, mock_asr, event_bus):
    """Long utterance (10 words) at conf=0.20 is accepted (>= 8 words bypasses short filter)."""
    long_text = "I would really like to know what time dinner is served tonight"
    mock_asr.transcribe.return_value = _make_result(long_text, 0.20)
    accepted = _collect_events(event_bus, EventType.TRANSCRIPT_READY)
    rejected = _collect_events(event_bus, EventType.TRANSCRIPT_REJECTED)

    result = await transcriber.process_separated_audio(_separation())

    assert result is not None, "Long utterance should be accepted even at low confidence"
    assert len(accepted) == 1
    assert len(rejected) == 0


# ---------------------------------------------------------------------------
# Edge cases: empty, ASR unavailable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_transcript_rejected(transcriber, mock_asr, event_bus):
    """Empty text is rejected with reason 'empty_transcript'."""
    mock_asr.transcribe.return_value = _make_result("", 0.80)
    rejected = _collect_events(event_bus, EventType.TRANSCRIPT_REJECTED)

    result = await transcriber.process_separated_audio(_separation())

    assert result is None
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "empty_transcript"


@pytest.mark.asyncio
async def test_asr_unavailable_rejected(transcriber, mock_asr, event_bus):
    """ASR returning None is rejected with reason 'asr_unavailable'."""
    mock_asr.transcribe.return_value = None
    rejected = _collect_events(event_bus, EventType.TRANSCRIPT_REJECTED)

    result = await transcriber.process_separated_audio(_separation())

    assert result is None
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "asr_unavailable"


# ---------------------------------------------------------------------------
# NeMo confidence extraction tests
# ---------------------------------------------------------------------------

def test_extract_confidence_word_confidence():
    """_extract_confidence uses min(word_confidence) from NeMo Hypothesis."""
    asr = ParakeetASR.__new__(ParakeetASR)
    asr._config = Config().asr
    asr._model = None
    asr._available = False

    hyp = MagicMock()
    hyp.word_confidence = [0.9, 0.7, 0.8]

    conf = asr._extract_confidence(hyp)

    assert conf == pytest.approx(0.7), f"Expected 0.7 (min of word_confidence), got {conf}"


def test_extract_confidence_fallback():
    """_extract_confidence falls back to 0.65 when word_confidence is absent."""
    asr = ParakeetASR.__new__(ParakeetASR)
    asr._config = Config().asr
    asr._model = None
    asr._available = False

    # Hypothesis with no word_confidence attribute
    hyp = MagicMock(spec=[])  # spec=[] means no attributes

    conf = asr._extract_confidence(hyp)

    assert conf == pytest.approx(0.65), f"Expected 0.65 fallback, got {conf}"


# ---------------------------------------------------------------------------
# Expanded blocklist
# ---------------------------------------------------------------------------

def test_expanded_blocklist_subscribers():
    """HALLUCINATION_PHRASES includes 'subscribers'."""
    assert "subscribers" in HALLUCINATION_PHRASES, (
        "'subscribers' should be in HALLUCINATION_PHRASES"
    )


def test_expanded_blocklist_like_and_subscribe():
    """HALLUCINATION_PHRASES includes 'like and subscribe'."""
    assert "like and subscribe" in HALLUCINATION_PHRASES, (
        "'like and subscribe' should be in HALLUCINATION_PHRASES"
    )
