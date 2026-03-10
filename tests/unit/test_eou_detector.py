"""Test scaffold for EOUDetector — extended with VAD-based EOU tests (Plan 05-01).

Heuristic predict tests PASS.
VAD-based tests cover feed_vad_prob() with 1800ms silence threshold.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from smait.core.config import Config
from smait.core.events import EventBus
from smait.perception.eou_detector import EOUDetector


@pytest.fixture
def detector(config, event_bus):
    """EOUDetector without init_model() — _available stays False."""
    return EOUDetector(config, event_bus)


def test_heuristic_predict_empty(detector):
    """predict() returns 0.0 for empty/whitespace text."""
    assert detector.predict("") == 0.0
    assert detector.predict("   ") == 0.0


def test_heuristic_predict_question(detector):
    """Heuristic returns ~0.9 for questions ending with '?'."""
    score = detector.predict("How are you today?")
    assert score == pytest.approx(0.9), f"Expected 0.9 for question, got {score}"


def test_heuristic_predict_statement(detector):
    """Heuristic returns ~0.8 for statements ending with '.' or '!'."""
    score_period = detector.predict("I would like some coffee.")
    assert score_period == pytest.approx(0.8), f"Expected 0.8 for period, got {score_period}"

    score_exclaim = detector.predict("That is amazing!")
    assert score_exclaim == pytest.approx(0.8), f"Expected 0.8 for exclamation, got {score_exclaim}"


def test_heuristic_predict_short_phrase(detector):
    """Heuristic returns 0.3 for single/double word utterances."""
    score = detector.predict("Yes")
    assert score == pytest.approx(0.3), f"Expected 0.3 for short phrase, got {score}"


def test_heuristic_predict_multi_word(detector):
    """Heuristic returns 0.6 for multi-word incomplete utterances."""
    score = detector.predict("I think maybe")
    assert score == pytest.approx(0.6), f"Expected 0.6 for multi-word, got {score}"


def test_predict_returns_float_in_range(detector):
    """predict() always returns a float in [0, 1] regardless of input."""
    texts = [
        "",
        "Hello",
        "How are you?",
        "This is a longer sentence that keeps going",
        "Done.",
        "WOW!",
    ]
    for text in texts:
        result = detector.predict(text)
        assert isinstance(result, float), f"Expected float for '{text}', got {type(result)}"
        assert 0.0 <= result <= 1.0, f"Expected [0, 1], got {result} for '{text}'"


def test_available_flag_starts_false(detector):
    """EOUDetector starts unavailable before init_model is called."""
    assert detector.available is False


def test_no_livekit_import():
    """QUAL-01: eou_detector.py source must not contain 'livekit' or 'turn_detector' strings."""
    source_path = pathlib.Path(__file__).parent.parent.parent / "smait" / "perception" / "eou_detector.py"
    source = source_path.read_text(encoding="utf-8")

    assert "livekit" not in source, (
        "eou_detector.py still references 'livekit'. "
        "Plan 03 must remove the LiveKit import and leave heuristic-only code."
    )
    assert "turn_detector" not in source, (
        "eou_detector.py still references 'turn_detector'."
    )
    assert "turn-detector" not in source, (
        "eou_detector.py still references 'turn-detector'."
    )


def test_no_transformers_import():
    """QUAL-01: eou_detector.py source must not contain 'transformers' or 'AutoModel' strings."""
    source_path = pathlib.Path(__file__).parent.parent.parent / "smait" / "perception" / "eou_detector.py"
    source = source_path.read_text(encoding="utf-8")

    assert "transformers" not in source, (
        "eou_detector.py still references 'transformers'. Plan 03 must remove it."
    )
    assert "AutoModel" not in source, (
        "eou_detector.py still references 'AutoModel'. Plan 03 must remove it."
    )


def test_init_model_sets_unavailable(config, event_bus):
    """After init_model(), _available is False (no model loaded in Phase 1)."""
    import asyncio
    detector = EOUDetector(config, event_bus)
    asyncio.run(detector.init_model())
    assert detector._available is False, (
        f"Expected _available=False after init_model(), got {detector._available}"
    )


def test_on_silence_hard_cutoff(config, event_bus):
    """on_silence triggers END_OF_TURN after hard_cutoff_ms of silence."""
    import asyncio
    emitted_events = []

    detector = EOUDetector(config, event_bus)
    asyncio.run(detector.init_model())

    # Subscribe to END_OF_TURN events
    from smait.core.events import EventType
    event_bus.subscribe(EventType.END_OF_TURN, lambda data: emitted_events.append(data))

    # Set up transcript and start silence timer
    t0 = 0.0
    detector.update_transcript("Hello there", timestamp=t0)

    # First on_silence call sets silence_start
    detector.on_silence(timestamp=t0 + 0.1)

    # Second on_silence call at t0 + hard_cutoff_s triggers hard cutoff
    # (silence_ms = (t0 + hard_cutoff_s) - silence_start = hard_cutoff_s - 0.1s + extra)
    hard_cutoff_s = config.eou.hard_cutoff_ms / 1000.0
    detector.on_silence(timestamp=t0 + 0.1 + hard_cutoff_s)

    assert len(emitted_events) == 1, (
        f"Expected 1 END_OF_TURN event from hard cutoff, got {len(emitted_events)}"
    )
    assert emitted_events[0]["reason"] == "hard_cutoff"


# ---------------------------------------------------------------------------
# VAD-based EOU tests (Phase 05-01: feed_vad_prob)
# ---------------------------------------------------------------------------

def _collect_eou_events(event_bus):
    """Helper: subscribe to END_OF_TURN and return the collecting list."""
    from smait.core.events import EventType
    events = []
    event_bus.subscribe(EventType.END_OF_TURN, lambda data: events.append(data))
    return events


def test_vad_silence_triggers_eou(config, event_bus):
    """feed_vad_prob: 1800ms of silence (28800 samples at 16kHz) after speech emits END_OF_TURN."""
    detector = EOUDetector(config, event_bus)
    events = _collect_eou_events(event_bus)

    # One chunk of high speech probability (sets _in_speech=True, _pending_turn=True)
    detector.feed_vad_prob(speech_prob=0.9, n_samples=480, timestamp=0.0)

    # Feed 60 chunks of 480 samples each at low prob (60 * 480 = 28800 samples = 1800ms at 16kHz)
    # All below 0.35 exit threshold
    for i in range(60):
        ts = (i + 1) * 0.03  # 30ms per chunk
        detector.feed_vad_prob(speech_prob=0.1, n_samples=480, timestamp=ts)

    assert len(events) == 1, f"Expected 1 END_OF_TURN, got {len(events)}"
    assert events[0]["reason"] == "vad_silence", f"Expected reason='vad_silence', got {events[0]}"


def test_vad_short_silence_no_eou(config, event_bus):
    """feed_vad_prob: short silence (33 chunks * 480 = 15840 < 28800) does NOT emit END_OF_TURN."""
    detector = EOUDetector(config, event_bus)
    events = _collect_eou_events(event_bus)

    # Speech start
    detector.feed_vad_prob(speech_prob=0.9, n_samples=480, timestamp=0.0)

    # Only 33 chunks of silence (15840 samples < 28800 threshold)
    for i in range(33):
        ts = (i + 1) * 0.03
        detector.feed_vad_prob(speech_prob=0.1, n_samples=480, timestamp=ts)

    assert len(events) == 0, f"Expected no END_OF_TURN for short silence, got {len(events)}"


def test_vad_speech_resets_counter(config, event_bus):
    """feed_vad_prob: speech mid-silence resets counter; exactly 1 END_OF_TURN on second full silence."""
    detector = EOUDetector(config, event_bus)
    events = _collect_eou_events(event_bus)

    # First speech burst
    detector.feed_vad_prob(speech_prob=0.9, n_samples=480, timestamp=0.0)

    # 20 chunks of silence (9600 samples, not enough to trigger)
    for i in range(20):
        detector.feed_vad_prob(speech_prob=0.1, n_samples=480, timestamp=(i + 1) * 0.03)

    # Speech again — resets silence counter
    detector.feed_vad_prob(speech_prob=0.8, n_samples=480, timestamp=0.63)

    # Now 60 more chunks of silence — should trigger EOU
    for i in range(60):
        ts = 0.66 + i * 0.03
        detector.feed_vad_prob(speech_prob=0.1, n_samples=480, timestamp=ts)

    assert len(events) == 1, f"Expected exactly 1 END_OF_TURN, got {len(events)}"


def test_vad_hysteresis_zone(config, event_bus):
    """feed_vad_prob: prob in hysteresis zone (0.35-0.5) does not change silence count."""
    detector = EOUDetector(config, event_bus)
    events = _collect_eou_events(event_bus)

    # Speech to enter _in_speech
    detector.feed_vad_prob(speech_prob=0.9, n_samples=480, timestamp=0.0)

    # Hysteresis zone — should neither increment silence nor reset counter
    for i in range(100):
        detector.feed_vad_prob(speech_prob=0.4, n_samples=480, timestamp=(i + 1) * 0.03)

    # No END_OF_TURN because silence samples never accumulated
    assert len(events) == 0, f"Expected no EOU in hysteresis zone, got {len(events)}"
    # And detector is still in speech state (_in_speech stays True, _silence_sample_count stays 0)
    assert detector._in_speech is True
    assert detector._silence_sample_count == 0


def test_vad_no_turn_without_speech(config, event_bus):
    """feed_vad_prob: silence without prior speech does NOT emit END_OF_TURN."""
    detector = EOUDetector(config, event_bus)
    events = _collect_eou_events(event_bus)

    # 60 chunks of silence right away (no speech first)
    for i in range(60):
        detector.feed_vad_prob(speech_prob=0.1, n_samples=480, timestamp=i * 0.03)

    assert len(events) == 0, f"Expected no EOU without prior speech, got {len(events)}"


def test_reset_clears_vad_state(config, event_bus):
    """reset() clears _in_speech, _silence_sample_count, and _pending_turn."""
    detector = EOUDetector(config, event_bus)

    # Set up an active turn with partial silence
    detector.feed_vad_prob(speech_prob=0.9, n_samples=480, timestamp=0.0)
    detector.feed_vad_prob(speech_prob=0.1, n_samples=480, timestamp=0.03)

    assert detector._in_speech is True or detector._silence_sample_count > 0 or detector._pending_turn is True

    detector.reset()

    assert detector._in_speech is False, f"Expected _in_speech=False after reset, got {detector._in_speech}"
    assert detector._silence_sample_count == 0, f"Expected _silence_sample_count=0 after reset, got {detector._silence_sample_count}"
    assert detector._pending_turn is False, f"Expected _pending_turn=False after reset, got {detector._pending_turn}"
