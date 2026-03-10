"""Tests for TTSEngine — Plan 02.

xfail markers removed; new tests added for voice config and sample rate.
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


def test_correct_class_imported(config, event_bus):
    """QUAL-01: init_model() must instantiate KPipeline(lang_code='a'), not KokoroTTS."""
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    with patch("kokoro.KPipeline", mock_pipeline_cls):
        engine = TTSEngine(config, event_bus)
        asyncio.run(engine.init_model())

    mock_pipeline_cls.assert_called_once_with(lang_code="a")
    assert engine.available is True


def test_synthesize_uses_generator(config, event_bus):
    """QUAL-01: synthesize() must consume KPipeline generator and concatenate audio chunks."""
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


def test_voice_from_config(config, event_bus):
    """Pipeline called with voice from config (default 'af_heart')."""
    audio_chunk = np.array([0.5], dtype=np.float32)

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = iter([(None, None, audio_chunk)])

    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

    with patch("kokoro.KPipeline", mock_pipeline_cls):
        engine = TTSEngine(config, event_bus)
        asyncio.run(engine.init_model())

    asyncio.run(engine.synthesize("Hello."))

    # Verify the pipeline was called with the correct voice
    mock_pipeline.assert_called_once()
    call_kwargs = mock_pipeline.call_args[1]
    assert "voice" in call_kwargs, "Expected 'voice' keyword argument in pipeline call"
    assert call_kwargs["voice"] == "af_heart", (
        f"Expected voice='af_heart', got voice='{call_kwargs['voice']}'"
    )


def test_sample_rate_24khz(config, event_bus):
    """Output PCM is at 24kHz — TTSConfig.sample_rate defaults to 24000."""
    engine = TTSEngine(config, event_bus)
    assert engine._sample_rate == 24000, (
        f"Expected sample_rate=24000, got {engine._sample_rate}"
    )


# ── New streaming and conversion tests (Plan 02-01) ──────────────────────────


def _make_engine_with_pipeline(config, event_bus, mock_pipeline):
    """Helper: create engine with _available=True and injected pipeline."""
    engine = TTSEngine(config, event_bus)
    engine._pipeline = mock_pipeline
    engine._available = True
    return engine


async def token_stream(tokens: list[str]):
    """Async generator that yields tokens one at a time."""
    for t in tokens:
        yield t


async def test_pcm_conversion_correct(config, event_bus):
    """synthesize() converts float32 audio to int16 PCM bytes with correct clipping.

    Values outside [-1, 1] must be clipped to [-32768, 32767].
    """
    # Include a value > 1.0 to exercise clipping
    audio_chunk = np.array([0.5, -0.5, 2.0, -2.0], dtype=np.float32)

    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda text, **kwargs: iter([(None, None, audio_chunk)])

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    result = await engine.synthesize("Test.")

    assert result is not None
    assert isinstance(result, bytes)
    pcm = np.frombuffer(result, dtype=np.int16)
    # 0.5 * 32767 ≈ 16383
    assert abs(int(pcm[0]) - 16383) <= 1, f"Expected ~16383, got {pcm[0]}"
    # -0.5 * 32767 ≈ -16383
    assert abs(int(pcm[1]) + 16383) <= 1, f"Expected ~-16383, got {pcm[1]}"
    # 2.0 clipped to 32767
    assert int(pcm[2]) == 32767, f"Expected 32767 (clipped), got {pcm[2]}"
    # -2.0 clipped to -32768
    assert int(pcm[3]) == -32768, f"Expected -32768 (clipped), got {pcm[3]}"


async def test_torch_tensor_handled(config, event_bus):
    """synthesize() calls .cpu().numpy() on torch-like tensors before PCM conversion."""
    audio_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Simulate a torch tensor with .cpu().numpy() interface
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.numpy.return_value = audio_array
    # Make hasattr(mock_tensor, 'cpu') return True — MagicMock supports attribute access by default

    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda text, **kwargs: iter([(None, None, mock_tensor)])

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    result = await engine.synthesize("Test.")

    assert result is not None, "Expected bytes result when tensor yields valid audio"
    assert isinstance(result, bytes)
    mock_tensor.cpu.assert_called_once()
    mock_tensor.cpu.return_value.numpy.assert_called_once()


async def test_none_audio_skipped(config, event_bus):
    """synthesize() skips chunks where audio is None (KPipeline quiet mode) without error."""
    audio_chunk = np.array([0.1, 0.2], dtype=np.float32)

    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda text, **kwargs: iter([
        (None, None, None),         # quiet chunk — must be skipped
        (None, None, audio_chunk),  # valid chunk
    ])

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    result = await engine.synthesize("Test.")

    assert result is not None, "Expected bytes from valid chunk after skipping None"
    assert isinstance(result, bytes)
    assert len(result) > 0


async def test_audio_chunk_is_bytes(config, event_bus):
    """synthesize() returns bytes — verifies the contract that ConnectionManager expects."""
    audio_chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda text, **kwargs: iter([(None, None, audio_chunk)])

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    result = await engine.synthesize("Test.")

    assert isinstance(result, bytes), (
        f"synthesize() must return bytes for ConnectionManager, got {type(result)}"
    )


async def test_streaming_event_order(config, event_bus):
    """speak_streaming() emits TTS_START before any TTS_AUDIO_CHUNK, TTS_END after all."""
    from smait.core.events import EventType

    order: list[str] = []

    async def on_start(_data):
        order.append("TTS_START")

    async def on_chunk(_data):
        order.append("TTS_AUDIO_CHUNK")

    async def on_end(_data):
        order.append("TTS_END")

    event_bus.subscribe(EventType.TTS_START, on_start)
    event_bus.subscribe(EventType.TTS_AUDIO_CHUNK, on_chunk)
    event_bus.subscribe(EventType.TTS_END, on_end)

    audio_chunk = np.array([0.1], dtype=np.float32)
    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda text, **kwargs: iter([(None, None, audio_chunk)])

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)

    # "Hello world. " has a sentence boundary after "world."
    await engine.speak_streaming(token_stream(["Hello world. "]))

    assert order[0] == "TTS_START", f"First event must be TTS_START, got: {order}"
    assert order[-1] == "TTS_END", f"Last event must be TTS_END, got: {order}"
    assert "TTS_AUDIO_CHUNK" in order, "At least one TTS_AUDIO_CHUNK expected"
    # All audio chunks must appear after TTS_START
    start_idx = order.index("TTS_START")
    end_idx = order.index("TTS_END")
    for i, ev in enumerate(order):
        if ev == "TTS_AUDIO_CHUNK":
            assert start_idx < i < end_idx, (
                f"TTS_AUDIO_CHUNK at position {i} is not between START({start_idx}) and END({end_idx})"
            )


async def test_streaming_emits_interleaved(config, event_bus):
    """speak_streaming() emits sentence 1 audio BEFORE synthesizing sentence 2."""
    from smait.core.events import EventType

    call_log: list[str] = []

    audio_chunk = np.array([0.1], dtype=np.float32)

    async def on_chunk(_data):
        call_log.append("EMIT_CHUNK")

    event_bus.subscribe(EventType.TTS_AUDIO_CHUNK, on_chunk)

    mock_pipeline = MagicMock()
    # Returns a fresh iterator each call (side_effect pattern from plan)
    call_counter = {"n": 0}

    def pipeline_side_effect(text, **kwargs):
        call_counter["n"] += 1
        call_log.append(f"SYNTHESIZE_{call_counter['n']}")
        return iter([(None, None, audio_chunk)])

    mock_pipeline.side_effect = pipeline_side_effect

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    # Two clear sentences — should each trigger a synthesize call
    await engine.speak_streaming(token_stream(["Hello world. ", "How are you? "]))

    # Verify interleaving: SYNTHESIZE_1, EMIT_CHUNK, SYNTHESIZE_2, EMIT_CHUNK
    assert "SYNTHESIZE_1" in call_log
    assert "SYNTHESIZE_2" in call_log
    s1_idx = call_log.index("SYNTHESIZE_1")
    s2_idx = call_log.index("SYNTHESIZE_2")
    # First EMIT_CHUNK after SYNTHESIZE_1 should be before SYNTHESIZE_2
    chunk_after_s1 = next(
        (i for i, e in enumerate(call_log) if e == "EMIT_CHUNK" and i > s1_idx), None
    )
    assert chunk_after_s1 is not None, "No EMIT_CHUNK found after SYNTHESIZE_1"
    assert chunk_after_s1 < s2_idx, (
        f"Audio chunk (pos {chunk_after_s1}) must be emitted before SYNTHESIZE_2 (pos {s2_idx}).\n"
        f"Actual log: {call_log}"
    )


async def test_streaming_flushes_remainder(config, event_bus):
    """speak_streaming() flushes remaining buffer text when LLM stream ends."""
    call_log: list[str] = []

    audio_chunk = np.array([0.1], dtype=np.float32)

    mock_pipeline = MagicMock()

    def pipeline_side_effect(text, **kwargs):
        call_log.append(text)
        return iter([(None, None, audio_chunk)])

    mock_pipeline.side_effect = pipeline_side_effect

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    # "Fine" has no trailing punctuation+space — must be flushed at stream end
    await engine.speak_streaming(token_stream(["Hello world. ", "Fine"]))

    assert any("Fine" in t for t in call_log), (
        f"Remaining text 'Fine' was not synthesized. Synthesize calls: {call_log}"
    )


async def test_sentence_splitting(config, event_bus):
    """_speak_by_sentence() splits text into 3 sentences and synthesizes each."""
    synthesize_calls: list[str] = []
    audio_chunk = np.array([0.1], dtype=np.float32)

    mock_pipeline = MagicMock()

    def pipeline_side_effect(text, **kwargs):
        synthesize_calls.append(text)
        return iter([(None, None, audio_chunk)])

    mock_pipeline.side_effect = pipeline_side_effect

    engine = _make_engine_with_pipeline(config, event_bus, mock_pipeline)
    await engine._speak_by_sentence("Hello world. How are you? Fine.")

    assert len(synthesize_calls) == 3, (
        f"Expected 3 synthesize calls, got {len(synthesize_calls)}: {synthesize_calls}"
    )
