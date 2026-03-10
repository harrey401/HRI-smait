"""Kokoro-82M TTS: sentence-level streaming, mic gating."""

from __future__ import annotations

import logging
import re
import time
from typing import AsyncGenerator, Optional

import numpy as np

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

# Sentence boundary detection pattern
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


class TTSEngine:
    """Kokoro-82M: <100ms TTFB, ~1GB VRAM, 96x real-time.

    Sentence-level streaming pipeline:
    1. Subscribe to DIALOGUE_STREAM events (partial LLM output)
    2. Buffer tokens until sentence boundary detected (.!?)
    3. Generate audio for that sentence with Kokoro
    4. Emit TTS_AUDIO_CHUNK -> ConnectionManager sends to Jackie
    5. Jackie plays audio through AudioTrack

    Mic gating:
    - Before first chunk: emit TTS_START -> Jackie gates mic
    - After last chunk: emit TTS_END -> Jackie ungates mic

    Fallback: if Kokoro fails, send text via JSON for Android TTS.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.tts
        self._event_bus = event_bus
        self._pipeline = None
        self._available = False
        self._sample_rate = config.tts.sample_rate
        self._voice = getattr(config.tts, "voice", "af_heart")

        # Streaming state
        self._text_buffer = ""
        self._is_speaking = False

    async def init_model(self) -> None:
        """Load Kokoro-82M TTS model."""
        logger.info("Loading Kokoro-82M TTS model...")
        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
            self._pipeline = KPipeline(lang_code="a")
            self._available = True
            logger.info(
                "Kokoro-82M loaded (voice=%s, sample_rate=%d)",
                self._voice,
                self._sample_rate,
            )
        except ImportError:
            logger.warning(
                "Kokoro TTS not installed. TTS will use Android fallback. "
                "Install with: pip install kokoro>=0.9.4 soundfile && sudo apt install espeak-ng"
            )
        except Exception:
            logger.exception("Failed to load Kokoro model")

    @property
    def available(self) -> bool:
        return self._available

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize a complete text string to PCM audio bytes.

        Returns PCM16 mono audio at 24kHz, or None if TTS is unavailable.
        """
        if not self._available or self._pipeline is None:
            return None

        try:
            t0 = time.monotonic()

            pcm_parts = []
            for _graphemes, _phonemes, audio in self._pipeline(
                text, voice=self._voice, speed=1.0
            ):
                # KPipeline quiet mode yields None audio — skip silently
                if audio is None:
                    continue
                # GPU path: KPipeline may yield torch tensors instead of numpy arrays
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                elif not isinstance(audio, np.ndarray):
                    audio = np.array(audio)
                # Convert float32 to int16 PCM with clipping
                pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                pcm_parts.append(pcm.tobytes())

            pcm_bytes = b"".join(pcm_parts)

            latency = (time.monotonic() - t0) * 1000
            logger.debug("TTS: '%s' -> %d bytes (%.1fms)",
                         text[:50], len(pcm_bytes), latency)
            return pcm_bytes

        except Exception:
            logger.exception("TTS synthesis failed")
            return None

    async def speak(self, text: str) -> None:
        """Speak a complete text with mic gating.

        For non-streaming use (e.g., greetings, farewell).
        """
        # Emit TTS_START for mic gating
        self._is_speaking = True
        await self._event_bus.emit_async(EventType.TTS_START)

        try:
            if self._config.stream_by_sentence:
                await self._speak_by_sentence(text)
            else:
                pcm = await self.synthesize(text)
                if pcm:
                    await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm})
                else:
                    # Fallback to Android TTS
                    await self._event_bus.emit_async(EventType.DIALOGUE_RESPONSE, {
                        "text": text,
                        "fallback_tts": True,
                    })
        finally:
            self._is_speaking = False
            await self._event_bus.emit_async(EventType.TTS_END)

    async def _speak_by_sentence(self, text: str) -> None:
        """Split text into sentences and synthesize each one."""
        sentences = SENTENCE_BOUNDARY.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return

        for sentence in sentences:
            pcm = await self.synthesize(sentence)
            if pcm:
                await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm})
            else:
                # Fallback for this sentence
                await self._event_bus.emit_async(EventType.DIALOGUE_RESPONSE, {
                    "text": sentence,
                    "fallback_tts": True,
                })

    async def speak_streaming(self, text_generator: AsyncGenerator[str, None]) -> None:
        """Consume a streaming text generator and synthesize sentence-by-sentence.

        Used with DialogueManager.ask_streaming() for real-time TTS:
        - Buffer tokens from LLM
        - When sentence boundary detected -> synthesize that sentence
        - Stream audio to Jackie while LLM continues generating
        """
        self._is_speaking = True
        await self._event_bus.emit_async(EventType.TTS_START)

        self._text_buffer = ""

        try:
            async for chunk in text_generator:
                self._text_buffer += chunk

                # Check for sentence boundaries in buffer
                while True:
                    match = SENTENCE_BOUNDARY.search(self._text_buffer)
                    if not match:
                        break

                    # Extract complete sentence
                    sentence = self._text_buffer[:match.start()].strip()
                    self._text_buffer = self._text_buffer[match.end():]

                    if sentence:
                        pcm = await self.synthesize(sentence)
                        if pcm:
                            await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm})

            # Flush remaining text
            remaining = self._text_buffer.strip()
            if remaining:
                pcm = await self.synthesize(remaining)
                if pcm:
                    await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm})
            self._text_buffer = ""

        finally:
            self._is_speaking = False
            await self._event_bus.emit_async(EventType.TTS_END)

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
