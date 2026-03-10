"""Test scaffold for TTSEngine — RED phase (Plan 01).

Passthrough/fallback tests PASS now.
Import-correctness tests are marked xfail until Plan 02 fixes the stub.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smait.core.config import Config
from smait.core.events import EventBus
from smait.output.tts import TTSEngine


@pytest.fixture
def tts_engine(config, event_bus):
    """TTSEngine without init_model() — _available stays False."""
    return TTSEngine(config, event_bus)


def test_unavailable_synthesize_returns_none(tts_engine):
    """Synthesize returns None when model has not been initialized."""
    result = asyncio.run(tts_engine.synthesize("Hello, world!"))

    assert result is None, "Expected None when TTS model is unavailable"


def test_available_flag_starts_false(tts_engine):
    """TTSEngine starts unavailable before init_model is called."""
    assert tts_engine.available is False


@pytest.mark.xfail(reason="Stub not yet fixed - Plan 02: tts.py still uses KokoroTTS")
def test_correct_class_imported(config, event_bus):
    """QUAL-01: init_model() must instantiate KPipeline(lang_code='a'), not KokoroTTS.

    This test is RED — it will fail until Plan 02 rewrites tts.py
    to use 'from kokoro import KPipeline'.
    """
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    with patch("kokoro.KPipeline", mock_pipeline_cls):
        engine = TTSEngine(config, event_bus)
        asyncio.run(engine.init_model())

    mock_pipeline_cls.assert_called_once_with(lang_code="a")
    assert engine.available is True


@pytest.mark.xfail(reason="Stub not yet fixed - Plan 02: tts.py synthesize() doesn't use KPipeline generator")
def test_synthesize_uses_generator(config, event_bus):
    """QUAL-01: synthesize() must consume KPipeline generator and concatenate audio chunks.

    This test is RED — it will fail until Plan 02 rewrites synthesize() to use
    'for gs, ps, audio in pipeline(text, voice=...)'.
    """
    import io

    audio_chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = iter([(None, None, audio_chunk)])

    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

    with patch("kokoro.KPipeline", mock_pipeline_cls):
        engine = TTSEngine(config, event_bus)
        asyncio.run(engine.init_model())

    result = asyncio.run(engine.synthesize("Test sentence."))

    assert result is not None, "Expected PCM bytes, got None"
    assert isinstance(result, bytes), "Expected bytes output"
    assert len(result) > 0, "Expected non-empty audio bytes"
