"""Unit tests for SoftwareAEC and AudioPipeline AEC integration.

Tests cover:
- SoftwareAEC frame size constants (256 samples / 512 bytes)
- feed_far() buffers far-end audio correctly
- process_near() with matched near/far frames returns same-length bytes
- process_near() with no far-end returns empty bytes (nothing to cancel)
- Partial frames buffered until full frame available
- Passthrough when speexdsp is unavailable (graceful degradation)
- AudioPipeline AEC gating: skipped when cae_status.aec=True
- AudioPipeline AEC integration: process_near() called when cae_status.aec=False
"""

from __future__ import annotations

import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.sensors.audio_pipeline import AudioPipeline


# ---------------------------------------------------------------------------
# SoftwareAEC import helper (avoids import-time speexdsp failure)
# ---------------------------------------------------------------------------

def _make_aec_with_mock_ec() -> object:
    """Create SoftwareAEC with a mocked speexdsp.EchoCanceller.

    Patches 'speexdsp' at import time so SoftwareAEC initialises with
    self._available = True and a MagicMock echo canceller.
    """
    mock_ec_instance = MagicMock()
    # ec.process(near, far) returns same-length bytes
    mock_ec_instance.process.side_effect = lambda near, far: near

    mock_ec_class = MagicMock()
    mock_ec_class.create.return_value = mock_ec_instance

    mock_speexdsp = MagicMock()
    mock_speexdsp.EchoCanceller = mock_ec_class

    # Patch speexdsp in sys.modules before importing SoftwareAEC
    with patch.dict(sys.modules, {"speexdsp": mock_speexdsp}):
        from smait.sensors.aec import SoftwareAEC  # noqa: PLC0415
        aec = SoftwareAEC(sample_rate=16000)

    # Detach mock from context: aec._ec is still the mock instance
    return aec


# ---------------------------------------------------------------------------
# Test 1: Frame size constants
# ---------------------------------------------------------------------------

def test_frame_size_constants():
    """SoftwareAEC.FRAME_SAMPLES == 256 and FRAME_BYTES == 512."""
    with patch.dict(sys.modules, {"speexdsp": MagicMock()}):
        from smait.sensors.aec import SoftwareAEC  # noqa: PLC0415
        assert SoftwareAEC.FRAME_SAMPLES == 256
        assert SoftwareAEC.FRAME_BYTES == 512


# ---------------------------------------------------------------------------
# Test 2: feed_far() buffers far-end audio
# ---------------------------------------------------------------------------

def test_feed_far_buffers():
    """feed_far() accumulates bytes in _far_buf."""
    aec = _make_aec_with_mock_ec()
    # Feed 1024 bytes far-end audio
    aec.feed_far(b"\x00" * 1024)
    assert len(aec._far_buf) == 1024


# ---------------------------------------------------------------------------
# Test 3: process_near() with matched near+far returns same-length output
# ---------------------------------------------------------------------------

def test_process_near_returns_bytes():
    """process_near() with one full near+far frame returns 512 bytes.

    Mock ec.process(near, far) to return near (passthrough).
    """
    aec = _make_aec_with_mock_ec()
    frame_bytes = 512  # FRAME_BYTES

    # Feed 512 bytes far-end
    far_data = b"\x01" * frame_bytes
    aec.feed_far(far_data)

    # Process 512 bytes near-end
    near_data = b"\x02" * frame_bytes
    output = aec.process_near(near_data)

    assert isinstance(output, bytes), f"Expected bytes, got {type(output)}"
    assert len(output) == frame_bytes, f"Expected {frame_bytes} bytes, got {len(output)}"


# ---------------------------------------------------------------------------
# Test 4: process_near() with no far-end returns empty bytes
# ---------------------------------------------------------------------------

def test_no_far_returns_empty():
    """process_near() with no far-end audio returns empty bytes (nothing to cancel)."""
    aec = _make_aec_with_mock_ec()

    # No feed_far() call — _far_buf is empty
    near_data = b"\x02" * 512
    output = aec.process_near(near_data)

    assert isinstance(output, bytes), f"Expected bytes, got {type(output)}"
    assert len(output) == 0, f"Expected empty output with no far-end, got {len(output)} bytes"


# ---------------------------------------------------------------------------
# Test 5: Partial frames buffered until full frame available
# ---------------------------------------------------------------------------

def test_partial_frames_buffered():
    """Partial frames are buffered; output only produced when full frame available.

    Feed 300 bytes near + 512 far → output empty (300 < 512).
    Feed another 212 bytes near → now 512 near bytes total → output = 512 bytes.
    """
    aec = _make_aec_with_mock_ec()
    frame_bytes = 512

    # Feed full far-end
    aec.feed_far(b"\x01" * frame_bytes)

    # Feed partial near (300 bytes < 512 frame)
    output1 = aec.process_near(b"\x02" * 300)
    assert len(output1) == 0, f"Expected empty output for partial frame, got {len(output1)}"
    assert len(aec._near_buf) == 300, f"Expected 300 bytes buffered, got {len(aec._near_buf)}"

    # Feed remaining 212 bytes → completes the 512-byte frame
    output2 = aec.process_near(b"\x03" * 212)
    assert len(output2) == frame_bytes, f"Expected {frame_bytes} bytes after completing frame, got {len(output2)}"


# ---------------------------------------------------------------------------
# Test 6: Passthrough when speexdsp unavailable
# ---------------------------------------------------------------------------

def test_passthrough_when_unavailable():
    """SoftwareAEC returns input unchanged when speexdsp is not installed.

    Simulate ImportError by patching speexdsp to raise on import.
    """
    # Remove speexdsp from sys.modules and patch to raise ImportError
    mock_bad_speexdsp = None  # will be absent from sys.modules

    # Use a fresh import by removing cached module
    aec_module_name = "smait.sensors.aec"
    if aec_module_name in sys.modules:
        del sys.modules[aec_module_name]

    # Patch speexdsp to raise ImportError
    with patch.dict(sys.modules, {"speexdsp": None}):
        from smait.sensors.aec import SoftwareAEC  # noqa: PLC0415
        aec = SoftwareAEC(sample_rate=16000)

    assert not aec.available, "Expected available=False when speexdsp missing"

    # process_near should return input unchanged (passthrough)
    near_data = b"\x02" * 512
    output = aec.process_near(near_data)
    assert output == near_data, "Expected passthrough output when unavailable"


# ---------------------------------------------------------------------------
# Test 7: AudioPipeline AEC gating — skipped when cae_status.aec=True
# ---------------------------------------------------------------------------

def test_cae_aec_gating_hardware_aec_active(config, event_bus):
    """When cae_status.aec=True, software AEC is NOT called.

    Mock SoftwareAEC with available=True, set cae_status.aec=True,
    call process_cae_audio(), assert _aec.process_near() was NOT called.
    """
    pipeline = AudioPipeline(config, event_bus)

    # Inject mocked SoftwareAEC
    mock_aec = MagicMock()
    mock_aec.available = True
    mock_aec.process_near.return_value = b""
    pipeline._aec = mock_aec

    # Set hardware AEC as active
    pipeline._cae_status = {"aec": True, "beamforming": False, "noise_suppression": False}

    # Mock VAD model
    mock_vad = MagicMock()
    mock_vad.return_value.item.return_value = 0.1  # silence
    pipeline._vad_model = mock_vad

    # Feed audio
    audio_chunk = np.zeros(480, dtype=np.int16).tobytes()
    pipeline.process_cae_audio(audio_chunk, 0.0)

    # Assert software AEC was NOT called (hardware AEC handles it)
    mock_aec.process_near.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: AudioPipeline AEC integration — process_near() called when cae_status.aec=False
# ---------------------------------------------------------------------------

def test_pipeline_aec_processes_audio(config, event_bus):
    """When cae_status.aec=False, software AEC process_near() is called.

    Mock SoftwareAEC with available=True and cae_status.aec=False.
    Assert _aec.process_near() was called with the audio data.
    """
    pipeline = AudioPipeline(config, event_bus)

    # Inject mocked SoftwareAEC
    audio_chunk = np.zeros(480, dtype=np.int16).tobytes()
    mock_aec = MagicMock()
    mock_aec.available = True
    mock_aec.process_near.return_value = audio_chunk  # return same data
    pipeline._aec = mock_aec

    # Set hardware AEC as NOT active (software AEC should engage)
    pipeline._cae_status = {"aec": False, "beamforming": False, "noise_suppression": False}

    # Mock VAD model
    mock_vad = MagicMock()
    mock_vad.return_value.item.return_value = 0.1  # silence
    pipeline._vad_model = mock_vad

    # Feed audio
    pipeline.process_cae_audio(audio_chunk, 0.0)

    # Assert software AEC WAS called
    mock_aec.process_near.assert_called_once_with(audio_chunk)
