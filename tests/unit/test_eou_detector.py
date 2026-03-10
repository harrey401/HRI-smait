"""Test scaffold for EOUDetector — RED phase (Plan 01).

Heuristic predict tests PASS now.
LiveKit import removal test is marked xfail until Plan 03 fixes eou_detector.py.
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
