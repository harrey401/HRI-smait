"""Test scaffold for DolphinSeparator — RED phase (Plan 01).

Passthrough/fallback tests PASS now.
Import-correctness tests are marked xfail until Plan 02 fixes the stub.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from smait.core.config import Config
from smait.core.events import EventBus
from smait.perception.dolphin_separator import DolphinSeparator


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


@pytest.mark.xfail(reason="Stub not yet fixed - Plan 02: dolphin_separator.py still uses DolphinModel")
def test_correct_import_path(config, event_bus):
    """QUAL-01: init_model() must call Dolphin.from_pretrained, not DolphinModel.

    This test is RED — it will fail until Plan 02 rewrites dolphin_separator.py
    to use 'from look2hear.models import Dolphin'.
    """
    mock_dolphin_cls = MagicMock()
    mock_model = MagicMock()
    mock_dolphin_cls.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model

    with patch("look2hear.models.Dolphin", mock_dolphin_cls):
        sep = DolphinSeparator(config, event_bus)
        asyncio.run(sep.init_model())

    mock_dolphin_cls.from_pretrained.assert_called_once_with("JusperLee/Dolphin")
