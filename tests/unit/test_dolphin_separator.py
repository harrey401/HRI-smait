"""Tests for DolphinSeparator — Plans 02 and 04-01.

Tests verifying correct imports and tensor shapes are now GREEN.
xfail markers removed after stub fixed.
Plan 04-01 adds: passthrough-on-empty-lip-frames, exception fallback, real output shape,
and inference_mode usage.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from smait.core.config import Config
from smait.core.events import EventBus
from smait.perception.dolphin_separator import DolphinSeparator
from smait.perception.lip_extractor import LipROI


@pytest.fixture
def separator(config, event_bus):
    """DolphinSeparator without init_model() — _available stays False."""
    return DolphinSeparator(config, event_bus)


def test_look2hear_importable():
    """ENV-03: from look2hear.models import Dolphin succeeds on CPU without weights."""
    from look2hear.models import Dolphin  # noqa: F401 — import is the test


def test_passthrough_returns_mono(separator, silence_audio):
    """Passthrough path returns float32 mono with confidence 0.0 when model unavailable."""
    result = asyncio.run(separator.separate(silence_audio, lip_frames=[], channels=1))

    assert result.separated_audio.dtype == np.float32, "Expected float32 output"
    assert result.separated_audio.ndim == 1, "Expected mono (1D) array"
    assert result.separation_confidence == 0.0, "Expected 0.0 confidence for passthrough"
    assert result.used_multichannel is False, "Expected False for single-channel passthrough"


def test_passthrough_mixes_multichannel(separator):
    """Passthrough with 4-channel audio mixes down to mono."""
    # 4-channel interleaved: 1000 frames * 4 channels = 4000 samples
    audio = np.ones(4000, dtype=np.int16) * 1000
    result = asyncio.run(separator.separate(audio, lip_frames=[], channels=4))

    assert result.separated_audio.ndim == 1, "Expected mono output"
    assert len(result.separated_audio) == 1000, "Expected 1000 mono samples"
    assert result.separation_confidence == 0.0


def test_correct_import_path(config, event_bus):
    """QUAL-01: init_model() must call Dolphin.from_pretrained, not DolphinModel.

    Patching look2hear.models.Dolphin verifies the stub now uses the correct import.
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    mock_dolphin_cls.from_pretrained.assert_called_once_with("JusperLee/Dolphin")


def test_run_dolphin_audio_shape(config, event_bus):
    """QUAL-01: _run_dolphin() passes audio tensor of shape [1, samples] (2D, not 3D).

    Dolphin takes mono audio — shape must be (batch=1, samples).
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    # Model returns 3D output matching real Dolphin: audio.unsqueeze(dim=1) = [1, 1, samples]
    import torch
    fake_output = torch.zeros(1, 1, 16000)
    mock_model.return_value = fake_output

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    # Prepare mono audio, with lip_frames so _run_dolphin actually calls the model
    T = 3
    lip_frames = [
        LipROI(
            image=np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
            timestamp=float(i) / 15.0,
            track_id=1,
        )
        for i in range(T)
    ]
    audio = np.zeros(16000, dtype=np.int16)
    asyncio.run(sep._run_dolphin(audio, lip_frames=lip_frames, channels=1, start=0.0))

    # Capture the actual audio tensor passed to the model
    call_args = mock_model.call_args
    assert call_args is not None, "Model was not called"
    audio_tensor = call_args[0][0]  # First positional arg

    assert audio_tensor.ndim == 2, f"Expected 2D audio tensor [1, samples], got {audio_tensor.ndim}D"
    assert audio_tensor.shape[0] == 1, f"Expected batch=1, got shape {audio_tensor.shape}"


def test_run_dolphin_video_shape(config, event_bus):
    """QUAL-01: _run_dolphin() passes video tensor of shape [1, 1, T, 88, 88, 1] (6D).

    Dolphin expects grayscale lip frames in (batch, 1, T, H=88, W=88, C=1) format.
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    # Model returns 3D output matching real Dolphin: audio.unsqueeze(dim=1) = [1, 1, samples]
    import torch
    fake_output = torch.zeros(1, 1, 16000)
    mock_model.return_value = fake_output

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    # Build 5 lip frames (RGB 96x96 crops, as from LipROI)
    T = 5
    lip_frames = []
    for i in range(T):
        roi = LipROI(
            image=np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
            timestamp=float(i) / 15.0,
            track_id=1,
        )
        lip_frames.append(roi)

    audio = np.zeros(16000, dtype=np.int16)
    asyncio.run(sep._run_dolphin(audio, lip_frames=lip_frames, channels=1, start=0.0))

    call_args = mock_model.call_args
    assert call_args is not None, "Model was not called"

    # Model called with (audio_tensor, video_tensor)
    assert len(call_args[0]) >= 2, "Expected at least 2 positional args (audio, video)"
    video_tensor = call_args[0][1]

    assert video_tensor.ndim == 6, (
        f"Expected 6D video tensor [1, 1, T, 88, 88, 1], got {video_tensor.ndim}D "
        f"shape={tuple(video_tensor.shape)}"
    )
    assert video_tensor.shape[0] == 1, "Expected batch=1"
    assert video_tensor.shape[1] == 1, "Expected second dim=1 (grayscale channel group)"
    assert video_tensor.shape[2] == T, f"Expected T={T} frames"
    assert video_tensor.shape[3] == 88, "Expected height=88"
    assert video_tensor.shape[4] == 88, "Expected width=88"
    assert video_tensor.shape[5] == 1, "Expected channel=1 (grayscale)"


# --- Plan 04-01 tests ---


def test_separate_without_lip_frames_uses_passthrough(config, event_bus):
    """SEP-01: When lip_frames=[], separate() returns passthrough WITHOUT calling Dolphin model.

    Dolphin.forward() requires both audio and video tensors — calling it without video
    raises TypeError. The early exit prevents this crash path.
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    audio = np.zeros(16000, dtype=np.int16)
    result = asyncio.run(sep.separate(audio, lip_frames=[], channels=1))

    # Passthrough: confidence 0.0, model NOT called
    assert result.separation_confidence == 0.0, (
        "Expected passthrough confidence=0.0 when lip_frames=[]"
    )
    assert mock_model.call_count == 0, (
        f"Expected model NOT to be called for empty lip_frames, but call_count={mock_model.call_count}"
    )


def test_dolphin_exception_falls_back_to_passthrough(config, event_bus):
    """SEP-03: When _run_dolphin raises RuntimeError, separate() catches it and returns passthrough."""
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    # Build lip frames so _run_dolphin would be reached (but we'll patch it to raise)
    lip_frames = [
        LipROI(
            image=np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
            timestamp=0.0,
            track_id=1,
        )
    ]
    audio = np.zeros(16000, dtype=np.int16)

    # Patch _run_dolphin on the instance to raise RuntimeError
    with patch.object(sep, "_run_dolphin", side_effect=RuntimeError("CUDA out of memory")):
        result = asyncio.run(sep.separate(audio, lip_frames=lip_frames, channels=1))

    assert result.separation_confidence == 0.0, (
        "Expected passthrough confidence=0.0 after _run_dolphin exception"
    )
    assert result.separated_audio.ndim == 1, "Expected 1D mono float32 passthrough output"


def test_run_dolphin_output_shape_matches_real_model(config, event_bus):
    """SEP-02: Mock model returns [1, 1, samples] (3D) matching real Dolphin forward().

    Real Dolphin.forward() returns audio.unsqueeze(dim=1) = [batch, 1, samples].
    After squeeze(), separated_audio must be 1D float32.
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    import torch
    # Real Dolphin output: [batch=1, 1, samples] — 3D
    mock_model.return_value = torch.zeros(1, 1, 16000)

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    lip_frames = [
        LipROI(
            image=np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
            timestamp=float(i) / 15.0,
            track_id=1,
        )
        for i in range(5)
    ]
    audio = np.zeros(16000, dtype=np.int16)
    result = asyncio.run(sep._run_dolphin(audio, lip_frames=lip_frames, channels=1, start=0.0))

    assert result.separated_audio.ndim == 1, (
        f"Expected 1D output after squeeze of [1,1,16000], got ndim={result.separated_audio.ndim}"
    )
    assert result.separated_audio.dtype == np.float32, (
        f"Expected float32, got {result.separated_audio.dtype}"
    )
    assert len(result.separated_audio) == 16000, (
        f"Expected 16000 samples, got {len(result.separated_audio)}"
    )


def test_inference_mode_used(config, event_bus):
    """SEP-04: _run_dolphin() must use torch.inference_mode(), not torch.no_grad().

    inference_mode is faster than no_grad (disables grad tracking entirely,
    skips view tracking overhead).
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    import torch
    mock_model.return_value = torch.zeros(1, 1, 16000)

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    lip_frames = [
        LipROI(
            image=np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
            timestamp=0.0,
            track_id=1,
        )
        for i in range(3)
    ]
    audio = np.zeros(16000, dtype=np.int16)

    with patch("torch.inference_mode") as mock_inference_mode:
        # inference_mode is a context manager — set up the mock properly
        mock_ctx = MagicMock()
        mock_inference_mode.return_value = mock_ctx
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        asyncio.run(sep._run_dolphin(audio, lip_frames=lip_frames, channels=1, start=0.0))

    mock_inference_mode.assert_called_once(), (
        "Expected torch.inference_mode() to be called once in _run_dolphin"
    )
