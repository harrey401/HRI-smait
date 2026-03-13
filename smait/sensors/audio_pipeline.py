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
from smait.sensors.aec import SoftwareAEC

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
    Implements barge-in detection: keeps VAD active during TTS playback
    and emits BARGE_IN when speech is detected, with a 200ms anti-echo guard.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.audio
        self._event_bus = event_bus

        # Barge-in state (replaces _mic_gated)
        self._tts_playing: bool = False
        self._tts_start_time: Optional[float] = None
        self._barge_in_min_speech_ms: int = config.eou.barge_in_min_speech_ms

        # CAE status (updated via CAE_STATUS events)
        self._cae_status: dict = {"aec": False, "beamforming": False, "noise_suppression": False}

        # Software AEC (fallback when hardware CAE AEC is not active)
        self._aec = SoftwareAEC(sample_rate=config.audio.sample_rate)

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
        event_bus.subscribe(EventType.CAE_STATUS, self._on_cae_status)
        event_bus.subscribe(EventType.TTS_AUDIO_CHUNK, self._on_tts_audio_chunk)

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
        """Process a chunk of CAE-processed audio through VAD.

        During TTS playback (_tts_playing=True):
        - VAD remains active to detect barge-in
        - If speech detected after 200ms anti-echo guard: emit BARGE_IN
        - No speech segment accumulation during TTS

        During normal listening (_tts_playing=False):
        - Normal VAD speech segment accumulation
        """
        if self._vad_model is None:
            return

        # Apply software AEC if hardware AEC is not active
        if self._aec.available and not self._cae_status.get("aec", False):
            processed = self._aec.process_near(data)
            if processed:
                data = processed

        # Convert bytes to float32 tensor for Silero
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_tensor = torch.from_numpy(samples)

        # Run VAD in 512-sample chunks (32ms at 16kHz) — Silero requires exactly 512
        chunk_size = 512
        offset = 0

        while offset + chunk_size <= len(chunk_tensor):
            chunk = chunk_tensor[offset:offset + chunk_size]
            speech_prob = self._vad_model(chunk, self._config.sample_rate).item()

            if self._tts_playing:
                # Barge-in detection mode: check for speech but don't accumulate segments
                if speech_prob >= self._vad_threshold:
                    # Apply anti-echo guard: ignore speech within barge_in_min_speech_ms of TTS_START
                    if self._tts_start_time is not None:
                        elapsed_ms = (time.monotonic() - self._tts_start_time) * 1000
                        if elapsed_ms < self._barge_in_min_speech_ms:
                            # Within echo guard window — skip
                            offset += chunk_size
                            continue

                    # Speech detected after guard window: emit BARGE_IN
                    logger.info("VAD: barge-in detected during TTS")
                    self._tts_playing = False
                    self._tts_start_time = None
                    self._event_bus.emit(EventType.BARGE_IN)
                    return  # Stop processing further chunks after barge-in
                # No speech during TTS: suppress segment accumulation
            else:
                # Normal listening mode: accumulate speech segments
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
        """Switch to barge-in detection mode during TTS playback."""
        self._tts_playing = True
        self._tts_start_time = time.monotonic()
        self._reset_speech()
        logger.debug("Barge-in detection active (TTS start)")

    def _on_tts_end(self, _data: object) -> None:
        """Return to normal listening mode after TTS playback."""
        self._tts_playing = False
        self._tts_start_time = None
        self._aec.reset()
        logger.debug("Normal listening mode (TTS end)")

    def _on_cae_status(self, data: object) -> None:
        """Update CAE status from hardware."""
        if isinstance(data, dict):
            self._cae_status = data
            logger.debug("CAE status updated: %s", data)

    def _on_tts_audio_chunk(self, data: object) -> None:
        """Feed TTS audio chunks to the software AEC as far-end reference.

        TTS outputs 24kHz PCM16 but AEC runs at 16kHz, so we resample
        the far-end reference before feeding it to the echo canceller.
        """
        if isinstance(data, dict):
            pcm_bytes = data.get("audio")
            if isinstance(pcm_bytes, (bytes, bytearray)):
                resampled = self._resample_24k_to_16k(pcm_bytes)
                self._aec.feed_far(resampled)
            elif hasattr(pcm_bytes, "tobytes"):
                resampled = self._resample_24k_to_16k(pcm_bytes.tobytes())
                self._aec.feed_far(resampled)

    @staticmethod
    def _resample_24k_to_16k(pcm_bytes: bytes) -> bytes:
        """Resample 24kHz PCM16 mono to 16kHz for AEC compatibility.

        Uses simple linear interpolation (2/3 ratio). For echo cancellation
        reference, this quality is sufficient — exact phase alignment matters
        more than pristine audio quality.
        """
        samples_24k = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n_out = int(len(samples_24k) * 16000 / 24000)
        if n_out == 0:
            return b""
        # Linear interpolation resampling
        x_old = np.linspace(0, 1, len(samples_24k))
        x_new = np.linspace(0, 1, n_out)
        samples_16k = np.interp(x_new, x_old, samples_24k)
        return samples_16k.astype(np.int16).tobytes()
