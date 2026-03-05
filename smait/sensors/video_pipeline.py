"""JPEG decode, frame buffer, and frame distribution."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

MAX_FRAME_BUFFER = 30  # Keep last N frames


@dataclass
class VideoFrame:
    """A decoded video frame with metadata."""
    image: np.ndarray       # BGR image (H, W, 3)
    timestamp: float        # Monotonic timestamp
    frame_id: int           # Sequential frame counter
    width: int
    height: int


class VideoPipeline:
    """Decodes JPEG frames from Jackie and distributes to consumers.

    Maintains a small frame buffer for temporal alignment with audio.
    Emits decoded frames for face tracking, gaze estimation, etc.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.vision
        self._event_bus = event_bus
        self._frame_counter = 0
        self._frame_buffer: deque[VideoFrame] = deque(maxlen=MAX_FRAME_BUFFER)
        self._latest_frame: Optional[VideoFrame] = None
        self._fps_counter = 0
        self._fps_time = time.monotonic()
        self._fps = 0.0

    def process_jpeg(self, jpeg_bytes: bytes, timestamp: float) -> Optional[VideoFrame]:
        """Decode a JPEG frame and distribute it."""
        try:
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            logger.warning("Failed to decode JPEG frame")
            return None

        if image is None:
            return None

        self._frame_counter += 1
        h, w = image.shape[:2]

        frame = VideoFrame(
            image=image,
            timestamp=timestamp,
            frame_id=self._frame_counter,
            width=w,
            height=h,
        )

        self._frame_buffer.append(frame)
        self._latest_frame = frame

        # FPS tracking
        self._fps_counter += 1
        elapsed = time.monotonic() - self._fps_time
        if elapsed >= 2.0:
            self._fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = time.monotonic()
            logger.debug("Video FPS: %.1f", self._fps)

        return frame

    @property
    def latest_frame(self) -> Optional[VideoFrame]:
        return self._latest_frame

    @property
    def fps(self) -> float:
        return self._fps

    def get_frame_at(self, timestamp: float, tolerance_ms: float = 100) -> Optional[VideoFrame]:
        """Get the frame closest to a given timestamp, within tolerance."""
        if not self._frame_buffer:
            return None

        best = None
        best_diff = float("inf")
        tol_s = tolerance_ms / 1000.0

        for frame in self._frame_buffer:
            diff = abs(frame.timestamp - timestamp)
            if diff < best_diff:
                best_diff = diff
                best = frame

        if best is not None and best_diff <= tol_s:
            return best
        return None

    def get_recent_frames(self, count: int) -> list[VideoFrame]:
        """Get the N most recent frames."""
        frames = list(self._frame_buffer)
        return frames[-count:] if len(frames) >= count else frames
