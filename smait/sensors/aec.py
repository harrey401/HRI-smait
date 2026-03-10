"""Software acoustic echo cancellation using speexdsp (graceful degradation).

Provides SoftwareAEC as a fallback when hardware AEC (CAE) is not active.
Uses frame-level processing with libspeexdsp via the speexdsp Python binding.

When speexdsp is not installed (requires libspeexdsp-dev + C compilation),
the class falls back to passthrough mode with a logged warning.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SoftwareAEC:
    """Frame-based software echo cancellation using speexdsp.

    Processes near-end (microphone) audio to remove echoes of far-end
    (TTS/speaker) audio that the microphone picks up.

    Frame size: 256 samples (512 bytes, int16 mono at 16kHz = 16ms).
    Far-end audio must be fed via feed_far() before process_near() can
    produce output — if no far-end reference is available, output is empty.

    Graceful degradation: if speexdsp is not installed, process_near()
    returns input unchanged (passthrough mode). Check .available to test.
    """

    FRAME_SAMPLES = 256
    FRAME_BYTES = FRAME_SAMPLES * 2  # int16 mono: 2 bytes per sample

    def __init__(self, sample_rate: int = 16000) -> None:
        self._near_buf = b""
        self._far_buf = b""

        try:
            from speexdsp import EchoCanceller  # type: ignore[import-not-found]
            self._ec = EchoCanceller.create(self.FRAME_SAMPLES, 2048, sample_rate)
            self._available = True
            logger.debug("SoftwareAEC: speexdsp EchoCanceller initialised (rate=%d)", sample_rate)
        except (ImportError, TypeError, AttributeError):
            logger.warning(
                "speexdsp not installed — software AEC unavailable. "
                "Install with: sudo apt install libspeexdsp-dev && pip install speexdsp"
            )
            self._ec = None
            self._available = False

    @property
    def available(self) -> bool:
        """True if speexdsp is installed and EchoCanceller is initialised."""
        return self._available

    def feed_far(self, pcm_bytes: bytes) -> None:
        """Buffer far-end (speaker/TTS) audio as echo reference.

        Call this with each TTS_AUDIO_CHUNK before the corresponding
        near-end audio arrives from the microphone.
        """
        self._far_buf += pcm_bytes

    def process_near(self, pcm_bytes: bytes) -> bytes:
        """Apply echo cancellation to near-end (microphone) audio.

        Processes complete 512-byte frames. Partial frames are buffered
        until enough data is available. Returns empty bytes if no far-end
        reference is available (far buffer too short for a full frame).

        If speexdsp is unavailable, returns input unchanged (passthrough).
        """
        if not self._available:
            return pcm_bytes

        self._near_buf += pcm_bytes
        output = b""

        while len(self._near_buf) >= self.FRAME_BYTES and len(self._far_buf) >= self.FRAME_BYTES:
            near_frame = self._near_buf[:self.FRAME_BYTES]
            far_frame = self._far_buf[:self.FRAME_BYTES]
            self._near_buf = self._near_buf[self.FRAME_BYTES:]
            self._far_buf = self._far_buf[self.FRAME_BYTES:]
            output += self._ec.process(near_frame, far_frame)

        return output

    def reset(self) -> None:
        """Clear internal near and far audio buffers.

        Call after TTS ends to discard stale reference audio.
        """
        self._near_buf = b""
        self._far_buf = b""
