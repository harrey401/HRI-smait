"""Unit tests for AudioPipeline VAD segmentation, ring buffer, and mic gating.

Tests cover:
- VAD emits SPEECH_SEGMENT after sufficient silence following speech
- Short speech segments (<0.5s) are rejected and not emitted
- RawAudioBuffer write and extract returns aligned audio
- RawAudioBuffer overrun returns None for stale data
- Mic gating suppresses VAD output during TTS playback
- _reset_speech guards against None vad_model (no AttributeError)
"""

from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.sensors.audio_pipeline import (
    AudioPipeline,
    RawAudioBuffer,
    SpeechSegment,
    MIN_SEGMENT_DURATION_S,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 480 samples


def _make_silence_chunk() -> bytes:
    """Generate silent int16 PCM audio (one 30ms chunk)."""
    return np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()


def _make_speech_chunk() -> bytes:
    """Generate noisy int16 PCM audio simulating speech (one 30ms chunk)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(CHUNK_SIZE) * 3000).astype(np.int16).tobytes()


def _make_pipeline(config: Config, event_bus: EventBus) -> AudioPipeline:
    """Create an AudioPipeline with a mocked VAD model (no torch.hub download)."""
    pipeline = AudioPipeline(config, event_bus)
    mock_vad = MagicMock()
    mock_vad.return_value.item.return_value = 0.1  # Default: silence
    pipeline._vad_model = mock_vad
    return pipeline


# ---------------------------------------------------------------------------
# VAD Segmentation Tests
# ---------------------------------------------------------------------------

def test_vad_emits_segment_after_silence(config, event_bus):
    """VAD emits SPEECH_SEGMENT after min_speech_duration_ms of silence following speech.

    Feed several speech chunks (prob >= threshold), then simulate silence long
    enough to trigger emission. Assert SPEECH_SEGMENT is emitted with a
    SpeechSegment containing cae_audio.
    """
    pipeline = _make_pipeline(config, event_bus)
    segments_received: list[SpeechSegment] = []
    event_bus.subscribe(EventType.SPEECH_SEGMENT, lambda seg: segments_received.append(seg))

    mock_vad = pipeline._vad_model

    # Feed speech chunks: probability >= 0.5 (vad_threshold)
    mock_vad.return_value.item.return_value = 0.9
    speech_chunk = _make_speech_chunk()
    t = 0.0
    # Feed ~1 second of speech (about 33 chunks of 30ms each)
    for i in range(33):
        t = i * (CHUNK_MS / 1000.0)
        pipeline.process_cae_audio(speech_chunk, t)

    # Now feed silence — mock monotonic to advance far enough past min_speech_duration_ms
    silence_chunk = _make_silence_chunk()
    mock_vad.return_value.item.return_value = 0.1

    # Patch time.monotonic so that the silence threshold is exceeded
    # min_speech_duration_ms=250 → need > 0.25s of monotonic time
    silence_start = time.monotonic()
    with patch("smait.sensors.audio_pipeline.time") as mock_time:
        # First silence call: sets silence_start_time
        mock_time.monotonic.return_value = silence_start
        pipeline.process_cae_audio(silence_chunk, t + CHUNK_MS / 1000.0)

        # Second silence call: 300ms elapsed → triggers emission
        mock_time.monotonic.return_value = silence_start + 0.3
        pipeline.process_cae_audio(silence_chunk, t + 2 * CHUNK_MS / 1000.0)

    assert len(segments_received) == 1, (
        f"Expected 1 SPEECH_SEGMENT, got {len(segments_received)}"
    )
    seg = segments_received[0]
    assert isinstance(seg, SpeechSegment)
    assert seg.cae_audio is not None
    assert len(seg.cae_audio) > 0
    assert seg.duration >= MIN_SEGMENT_DURATION_S


def test_short_segment_rejected(config, event_bus):
    """Short speech segments (<0.5s duration) are rejected and not emitted.

    Feed speech for only 0.1s then silence — the segment is shorter than
    MIN_SEGMENT_DURATION_S (0.5s), so no SPEECH_SEGMENT event is emitted.
    """
    pipeline = _make_pipeline(config, event_bus)
    segments_received: list[SpeechSegment] = []
    event_bus.subscribe(EventType.SPEECH_SEGMENT, lambda seg: segments_received.append(seg))

    mock_vad = pipeline._vad_model

    # Speech start at t=0.0
    mock_vad.return_value.item.return_value = 0.9
    speech_chunk = _make_speech_chunk()
    pipeline.process_cae_audio(speech_chunk, 0.0)

    # Only ~0.1s of speech — well below MIN_SEGMENT_DURATION_S (0.5s)
    # Now feed silence to trigger _emit_segment
    silence_chunk = _make_silence_chunk()
    mock_vad.return_value.item.return_value = 0.1

    silence_start = time.monotonic()
    with patch("smait.sensors.audio_pipeline.time") as mock_time:
        mock_time.monotonic.return_value = silence_start
        pipeline.process_cae_audio(silence_chunk, 0.03)

        mock_time.monotonic.return_value = silence_start + 0.3
        # timestamp is only 0.06s from start → duration = 0.06s < 0.5s → rejected
        pipeline.process_cae_audio(silence_chunk, 0.06)

    assert len(segments_received) == 0, (
        f"Expected 0 SPEECH_SEGMENTs (short segment), got {len(segments_received)}"
    )


def test_mic_gating_suppresses_vad(config, event_bus):
    """When mic is gated (TTS playing), VAD output is suppressed.

    Set _mic_gated=True, feed speech audio, assert no SPEECH_SEGMENT emitted.
    """
    pipeline = _make_pipeline(config, event_bus)
    segments_received: list[SpeechSegment] = []
    event_bus.subscribe(EventType.SPEECH_SEGMENT, lambda seg: segments_received.append(seg))

    # Gate the mic (simulates TTS_START event)
    pipeline._mic_gated = True

    mock_vad = pipeline._vad_model
    mock_vad.return_value.item.return_value = 0.9
    speech_chunk = _make_speech_chunk()

    for i in range(50):
        pipeline.process_cae_audio(speech_chunk, i * CHUNK_MS / 1000.0)

    assert len(segments_received) == 0, (
        f"Expected 0 SPEECH_SEGMENTs (mic gated), got {len(segments_received)}"
    )


def test_reset_speech_guards_vad_model_none(config, event_bus):
    """_reset_speech() does not raise AttributeError when _vad_model is None.

    The current code already has the guard; this test ensures it stays correct.
    """
    pipeline = AudioPipeline(config, event_bus)
    pipeline._vad_model = None
    pipeline._in_speech = True
    pipeline._speech_buffer = [_make_speech_chunk()]
    pipeline._speech_start_time = 0.0
    pipeline._silence_start_time = None

    # Should NOT raise
    pipeline._reset_speech()

    assert not pipeline._in_speech
    assert pipeline._speech_buffer == []
    assert pipeline._speech_start_time is None


# ---------------------------------------------------------------------------
# RawAudioBuffer Tests
# ---------------------------------------------------------------------------

def test_ring_buffer_write_and_extract(config):
    """Write timestamped audio to RawAudioBuffer, extract a window, verify content."""
    # Use a small mono buffer for simplicity (1 channel, 2 seconds)
    buf = RawAudioBuffer(sample_rate=16000, channels=1, buffer_seconds=2.0)

    # Write 1 second of data starting at t=0.0
    n_samples = 16000
    audio_data = (np.arange(n_samples, dtype=np.int16)).tobytes()
    buf.write(audio_data, timestamp=0.0)

    # Write another second starting at t=1.0
    audio_data2 = (np.arange(n_samples, n_samples * 2, dtype=np.int16)).tobytes()
    buf.write(audio_data2, timestamp=1.0)

    # Extract the first second: t=0.0 to t=1.0
    extracted = buf.extract(start_time=0.0, end_time=1.0)

    assert extracted is not None, "Expected extracted audio, got None"
    assert len(extracted) > 0, "Extracted audio should not be empty"
    # Extracted array should be int16 values from the first write
    assert extracted.dtype == np.int16


def test_ring_buffer_overrun_returns_none():
    """Requesting overwritten data from RawAudioBuffer returns None.

    Use a tiny buffer (0.1s) and write enough data to overwrite early data.
    Then attempt to extract from the early time window.
    """
    # Tiny buffer: 0.1 seconds of mono 16kHz = 1600 samples
    buf = RawAudioBuffer(sample_rate=16000, channels=1, buffer_seconds=0.1)

    # Write 0.2 seconds of data — this should overwrite the first 0.1s
    n_samples = 3200  # 0.2 seconds at 16kHz
    audio_data = np.zeros(n_samples, dtype=np.int16).tobytes()
    buf.write(audio_data, timestamp=0.0)

    # Attempt to extract from the beginning — data was overwritten
    result = buf.extract(start_time=0.0, end_time=0.05)

    # Should return None because the early data was overwritten
    assert result is None, (
        f"Expected None for overrun buffer, got array of length {len(result) if result is not None else 'N/A'}"
    )
