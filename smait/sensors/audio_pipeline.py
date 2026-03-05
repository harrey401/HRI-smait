"""Silero VAD + ring buffer → SpeechSegment production."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

# Minimum segment length to avoid Parakeet hallucinations (Issue #6)
MIN_SEGMENT_DURATION_S = 0.5


@dataclass
class SpeechSegment:
    """A complete speech segment with aligned audio from both streams."""
    cae_audio: np.ndarray          # CAE-processed single-channel (16kHz, int16)
    raw_audio: Optional[np.ndarray]  # Raw 4-channel (16kHz, int16, interleaved) or None
    start_time: float              # Monotonic timestamp
    end_time: float
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration = self.end_time - self.start_time


class RawAudioBuffer:
    """Time-aligned ring buffer for raw 4-channel audio.

    Stores raw multi-channel audio with timestamps so that when VAD
    identifies a speech segment in the CAE stream, the corresponding
    raw audio window can be extracted for Dolphin.
    """

    def __init__(self, sample_rate: int, channels: int, buffer_seconds: float) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._samples_per_channel = int(sample_rate * buffer_seconds)
        # Store as interleaved int16
        self._buffer = np.zeros(self._samples_per_channel * channels, dtype=np.int16)
        self._write_pos = 0
        self._timestamps: list[tuple[int, float]] = []  # (sample_offset, timestamp)
        self._total_samples_written = 0

    def write(self, data: bytes, timestamp: float) -> None:
        """Append raw audio data to the ring buffer."""
        samples = np.frombuffer(data, dtype=np.int16)
        n_samples = len(samples)

        if n_samples == 0:
            return

        buf_len = len(self._buffer)
        if self._write_pos + n_samples <= buf_len:
            self._buffer[self._write_pos:self._write_pos + n_samples] = samples
        else:
            first = buf_len - self._write_pos
            self._buffer[self._write_pos:] = samples[:first]
            self._buffer[:n_samples - first] = samples[first:]

        self._timestamps.append((self._total_samples_written, timestamp))
        self._write_pos = (self._write_pos + n_samples) % buf_len
        self._total_samples_written += n_samples

        # Prune old timestamps
        min_sample = self._total_samples_written - buf_len
        self._timestamps = [(s, t) for s, t in self._timestamps if s >= min_sample]

    def extract(self, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Extract raw audio for a time window. Returns None if not available."""
        if not self._timestamps:
            return None

        # Find the closest sample offsets for the requested time range
        start_offset = self._time_to_offset(start_time)
        end_offset = self._time_to_offset(end_time)

        if start_offset is None or end_offset is None:
            return None

        buf_len = len(self._buffer)
        total = self._total_samples_written

        # Convert absolute offsets to ring buffer positions
        start_pos = start_offset % buf_len
        n_samples = end_offset - start_offset

        # Check if the data is still in the buffer
        if total - start_offset > buf_len:
            logger.warning("Raw audio buffer overrun — requested data was overwritten")
            return None

        if n_samples <= 0 or n_samples > buf_len:
            return None

        if start_pos + n_samples <= buf_len:
            return self._buffer[start_pos:start_pos + n_samples].copy()
        else:
            first = buf_len - start_pos
            return np.concatenate([
                self._buffer[start_pos:],
                self._buffer[:n_samples - first],
            ])

    def _time_to_offset(self, t: float) -> Optional[int]:
        """Convert a monotonic timestamp to an approximate sample offset."""
        if not self._timestamps:
            return None

        # Find the two timestamps bracketing t
        prev = None
        for sample_offset, ts in self._timestamps:
            if ts >= t:
                if prev is None:
                    return sample_offset
                # Interpolate
                prev_offset, prev_ts = prev
                ratio = (t - prev_ts) / (ts - prev_ts) if ts != prev_ts else 0.0
                return int(prev_offset + ratio * (sample_offset - prev_offset))
            prev = (sample_offset, ts)

        # t is after all timestamps — extrapolate from last
        if prev:
            last_offset, last_ts = prev
            elapsed = t - last_ts
            return int(last_offset + elapsed * self._sample_rate * self._channels)
        return None


class AudioPipeline:
    """Receives CAE audio, runs Silero VAD, produces SpeechSegments.

    Also maintains the RawAudioBuffer for 4-channel audio alignment.
    Implements mic gating: disables VAD output during TTS playback.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.audio
        self._event_bus = event_bus
        self._mic_gated = False

        # Silero VAD
        self._vad_model: Optional[torch.nn.Module] = None
        self._vad_threshold = config.audio.vad_threshold

        # Speech accumulation
        self._speech_buffer: list[bytes] = []
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._in_speech = False

        # Raw audio ring buffer
        self._raw_buffer = RawAudioBuffer(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels_raw,
            buffer_seconds=config.audio.raw_buffer_seconds,
        )

        # Subscribe to events
        event_bus.subscribe(EventType.TTS_START, self._on_tts_start)
        event_bus.subscribe(EventType.TTS_END, self._on_tts_end)

    async def init_model(self) -> None:
        """Load Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._vad_model.eval()
        logger.info("Silero VAD loaded")

    def process_cae_audio(self, data: bytes, timestamp: float) -> None:
        """Process a chunk of CAE-processed audio through VAD."""
        if self._mic_gated or self._vad_model is None:
            return

        # Convert bytes to float32 tensor for Silero
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_tensor = torch.from_numpy(samples)

        # Run VAD on 30ms chunks (480 samples at 16kHz)
        chunk_size = int(self._config.sample_rate * self._config.chunk_duration_ms / 1000)
        offset = 0

        while offset + chunk_size <= len(chunk_tensor):
            chunk = chunk_tensor[offset:offset + chunk_size]
            speech_prob = self._vad_model(chunk, self._config.sample_rate).item()

            if speech_prob >= self._vad_threshold:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_start_time = timestamp
                    self._speech_buffer = []
                    self._silence_start_time = None
                    logger.debug("VAD: speech start")
                self._speech_buffer.append(data[offset * 2:(offset + chunk_size) * 2])
                self._silence_start_time = None
            else:
                if self._in_speech:
                    if self._silence_start_time is None:
                        self._silence_start_time = time.monotonic()
                    self._speech_buffer.append(data[offset * 2:(offset + chunk_size) * 2])

                    # Check if silence exceeds min_speech_duration threshold
                    silence_ms = (time.monotonic() - self._silence_start_time) * 1000
                    if silence_ms >= self._config.min_speech_duration_ms:
                        self._emit_segment(timestamp)

            offset += chunk_size

    def process_raw_audio(self, data: bytes, timestamp: float) -> None:
        """Store raw 4-channel audio in the ring buffer."""
        self._raw_buffer.write(data, timestamp)

    def _emit_segment(self, current_time: float) -> None:
        """Emit a complete speech segment."""
        if not self._speech_buffer or self._speech_start_time is None:
            return

        end_time = current_time
        duration = end_time - self._speech_start_time

        # Reject segments shorter than minimum (Issue #6 mitigation)
        if duration < MIN_SEGMENT_DURATION_S:
            logger.debug("VAD: rejecting short segment (%.2fs < %.2fs)",
                         duration, MIN_SEGMENT_DURATION_S)
            self._reset_speech()
            return

        # Combine CAE audio chunks
        cae_audio = np.frombuffer(b"".join(self._speech_buffer), dtype=np.int16)

        # Extract corresponding raw audio window
        raw_audio = self._raw_buffer.extract(self._speech_start_time, end_time)

        segment = SpeechSegment(
            cae_audio=cae_audio,
            raw_audio=raw_audio,
            start_time=self._speech_start_time,
            end_time=end_time,
        )

        logger.info("VAD: speech segment %.2fs (cae=%d samples, raw=%s)",
                     duration, len(cae_audio),
                     f"{len(raw_audio)} samples" if raw_audio is not None else "None")

        self._event_bus.emit(EventType.SPEECH_SEGMENT, segment)
        self._reset_speech()

    def _reset_speech(self) -> None:
        self._in_speech = False
        self._speech_buffer = []
        self._speech_start_time = None
        self._silence_start_time = None
        # Reset VAD state
        if self._vad_model is not None:
            self._vad_model.reset_states()

    def _on_tts_start(self, _data: object) -> None:
        """Gate microphone during TTS playback."""
        self._mic_gated = True
        if self._in_speech:
            self._reset_speech()
        logger.debug("Mic gated (TTS start)")

    def _on_tts_end(self, _data: object) -> None:
        """Ungate microphone after TTS playback."""
        self._mic_gated = False
        logger.debug("Mic ungated (TTS end)")
