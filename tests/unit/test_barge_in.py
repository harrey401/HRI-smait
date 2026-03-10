"""Unit tests for barge-in detection: BARGE_IN event, VAD during TTS, cancellable TTS.

Tests cover:
- BARGE_IN emitted when VAD detects speech during TTS playback
- No BARGE_IN without TTS playing (normal operation)
- Anti-echo delay guard: no BARGE_IN within 200ms of TTS_START
- TTSEngine._tts_task cancelled on BARGE_IN
- TTS_END emitted after barge-in cancellation (finally block)
- TTS natural completion still emits TTS_END
- AudioPipeline._tts_playing cleared after TTS_END event
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.sensors.audio_pipeline import AudioPipeline
from smait.output.tts import TTSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 480 samples


def _make_pipeline_with_vad(config: Config, event_bus: EventBus, speech_prob: float = 0.8) -> AudioPipeline:
    """Create AudioPipeline with a mock VAD returning a fixed speech probability.

    The Silero VAD is called as: speech_prob = vad_model(chunk, sample_rate).item()
    So mock_vad(chunk, rate) returns an object whose .item() returns the float.
    """
    pipeline = AudioPipeline(config, event_bus)
    mock_vad = MagicMock()
    mock_vad.return_value.item.return_value = speech_prob
    pipeline._vad_model = mock_vad
    return pipeline


def _make_audio_chunk() -> bytes:
    """Generate 480 int16 samples = 960 bytes (one 30ms chunk)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(CHUNK_SIZE) * 3000).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Test 1: BARGE_IN emitted when TTS is playing and speech detected
# ---------------------------------------------------------------------------

def test_barge_in_emitted_during_tts(config, event_bus):
    """When TTS is playing and VAD detects speech (prob >= 0.5), BARGE_IN is emitted.

    Simulate:
    1. Emit TTS_START to set _tts_playing=True and record _tts_start_time
    2. Advance time past 200ms anti-echo guard
    3. Call process_cae_audio() with speech prob 0.8
    4. Assert BARGE_IN event was emitted
    """
    pipeline = _make_pipeline_with_vad(config, event_bus, speech_prob=0.8)
    barge_in_received: list[object] = []
    event_bus.subscribe(EventType.BARGE_IN, lambda data: barge_in_received.append(data))

    # Simulate TTS_START
    event_bus.emit(EventType.TTS_START)

    audio_chunk = _make_audio_chunk()

    # Advance time past the 200ms barge-in guard
    with patch("smait.sensors.audio_pipeline.time") as mock_time:
        mock_time.monotonic.return_value = time.monotonic() + 0.5  # 500ms after start

        pipeline.process_cae_audio(audio_chunk, 0.0)

    assert len(barge_in_received) >= 1, (
        f"Expected BARGE_IN event, got {len(barge_in_received)}"
    )


# ---------------------------------------------------------------------------
# Test 2: No BARGE_IN without TTS playing
# ---------------------------------------------------------------------------

def test_no_barge_in_without_tts(config, event_bus):
    """When TTS is not playing, speech detection does NOT emit BARGE_IN.

    Normal operation: speech audio is processed normally but no barge-in event.
    """
    pipeline = _make_pipeline_with_vad(config, event_bus, speech_prob=0.8)
    barge_in_received: list[object] = []
    event_bus.subscribe(EventType.BARGE_IN, lambda data: barge_in_received.append(data))

    # Do NOT emit TTS_START — pipeline._tts_playing should be False

    audio_chunk = _make_audio_chunk()
    pipeline.process_cae_audio(audio_chunk, 0.0)

    assert len(barge_in_received) == 0, (
        f"Expected no BARGE_IN without TTS, got {len(barge_in_received)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Anti-echo delay guard blocks BARGE_IN within 200ms of TTS_START
# ---------------------------------------------------------------------------

def test_barge_in_delay_guard(config, event_bus):
    """No BARGE_IN is emitted within 200ms of TTS_START (anti-echo guard).

    Immediately after TTS_START, any speech detected is likely echo.
    The 200ms window prevents spurious barge-in events.
    """
    pipeline = _make_pipeline_with_vad(config, event_bus, speech_prob=0.8)
    barge_in_received: list[object] = []
    event_bus.subscribe(EventType.BARGE_IN, lambda data: barge_in_received.append(data))

    tts_start_epoch = time.monotonic()
    audio_chunk = _make_audio_chunk()

    with patch("smait.sensors.audio_pipeline.time") as mock_time:
        # Set TTS_START time
        mock_time.monotonic.return_value = tts_start_epoch
        event_bus.emit(EventType.TTS_START)

        # Call process_cae_audio at 50ms (within 200ms guard)
        mock_time.monotonic.return_value = tts_start_epoch + 0.05  # 50ms elapsed
        pipeline.process_cae_audio(audio_chunk, 0.0)

    assert len(barge_in_received) == 0, (
        f"Expected no BARGE_IN within 200ms guard, got {len(barge_in_received)}"
    )


# ---------------------------------------------------------------------------
# Test 4: BARGE_IN cancels _tts_task in TTSEngine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_task_cancelled_on_barge_in(config, event_bus):
    """When BARGE_IN is received by TTSEngine, the _tts_task is cancelled.

    Create a long-running task, assign to _tts_task, emit BARGE_IN, assert cancelled.
    """
    engine = TTSEngine(config, event_bus)

    # Create a long-running dummy task
    async def long_running():
        await asyncio.sleep(100)

    loop = asyncio.get_running_loop()
    dummy_task = loop.create_task(long_running())
    engine._tts_task = dummy_task

    # Emit BARGE_IN event synchronously (triggers _on_barge_in)
    event_bus.emit(EventType.BARGE_IN)

    # Give the event loop a chance to cancel the task
    await asyncio.sleep(0)

    assert dummy_task.cancelled() or dummy_task.done(), (
        "Expected TTS task to be cancelled after BARGE_IN"
    )

    # Cleanup
    if not dummy_task.done():
        dummy_task.cancel()
        try:
            await dummy_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Test 5: TTS_END emitted after barge-in cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_end_after_barge_in(config, event_bus):
    """TTS_END is always emitted after barge-in cancellation (finally block).

    Run speak_streaming() with a generator that never finishes.
    Cancel via BARGE_IN. Assert TTS_END was emitted.
    """
    engine = TTSEngine(config, event_bus)
    tts_end_received: list[object] = []
    event_bus.subscribe(EventType.TTS_END, lambda data: tts_end_received.append(data))

    # Async generator that yields one chunk then blocks
    async def blocking_generator():
        yield "Hello"
        await asyncio.sleep(100)

    async def run_tts():
        await engine.speak_streaming(blocking_generator())

    # Run TTS in a task so we can cancel it via barge-in
    tts_coroutine_task = asyncio.create_task(run_tts())

    # Wait a bit for TTS to start
    await asyncio.sleep(0.05)

    # Emit BARGE_IN to cancel
    event_bus.emit(EventType.BARGE_IN)

    # Wait for TTS task to finish (should be cancelled quickly)
    try:
        await asyncio.wait_for(tts_coroutine_task, timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    assert len(tts_end_received) >= 1, (
        f"Expected TTS_END after barge-in cancellation, got {len(tts_end_received)}"
    )


# ---------------------------------------------------------------------------
# Test 6: TTS completes naturally and still emits TTS_END
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_natural_completion(config, event_bus):
    """TTS completes naturally (no barge-in) and emits TTS_END.

    Run speak_streaming() with a simple generator. Assert TTS_END emitted
    and _is_speaking is False.
    """
    engine = TTSEngine(config, event_bus)
    tts_end_received: list[object] = []
    tts_start_received: list[object] = []
    event_bus.subscribe(EventType.TTS_END, lambda data: tts_end_received.append(data))
    event_bus.subscribe(EventType.TTS_START, lambda data: tts_start_received.append(data))

    # Mock synthesize to return immediately (no actual Kokoro model)
    engine._available = True
    engine.synthesize = AsyncMock(return_value=None)

    async def simple_generator():
        yield "Hello world."

    await engine.speak_streaming(simple_generator())

    assert len(tts_start_received) >= 1, "Expected TTS_START"
    assert len(tts_end_received) >= 1, "Expected TTS_END after natural completion"
    assert not engine.is_speaking, "Expected _is_speaking=False after completion"


# ---------------------------------------------------------------------------
# Test 7: _tts_playing cleared after TTS_END event
# ---------------------------------------------------------------------------

def test_tts_playing_cleared_after_end(config, event_bus):
    """AudioPipeline._tts_playing is False after TTS_END event is received.

    Emit TTS_START (sets _tts_playing=True), then TTS_END (should clear it).
    """
    pipeline = AudioPipeline(config, event_bus)

    # Emit TTS_START
    event_bus.emit(EventType.TTS_START)
    assert pipeline._tts_playing is True, "Expected _tts_playing=True after TTS_START"

    # Emit TTS_END
    event_bus.emit(EventType.TTS_END)
    assert pipeline._tts_playing is False, "Expected _tts_playing=False after TTS_END"
