"""Extract mouth ROI crops from face landmarks for Dolphin AV-TSE."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.face_tracker import FaceTrack

logger = logging.getLogger(__name__)

# MediaPipe lip landmark indices
OUTER_LIP_INDICES = list(range(61, 69))
INNER_LIP_INDICES = list(range(78, 96))
ALL_LIP_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES

# How many lip ROI frames to buffer per face
LIP_BUFFER_SECONDS = 5.0
LIP_BUFFER_FPS = 15  # Approximate frame rate for lip ROI buffer


@dataclass
class LipROI:
    """A mouth region crop with timestamp."""
    image: np.ndarray      # RGB crop, resized to lip_roi_size
    timestamp: float
    track_id: int


class LipExtractor:
    """Extracts mouth ROI crops from face tracker output for Dolphin.

    Uses MediaPipe landmarks 61-68 (outer lips) and 78-95 (inner lips)
    to compute a bounding box around the mouth, then crops and resizes.

    Maintains a temporal buffer of lip ROI frames per tracked face,
    synchronized with audio timestamps, for feeding to Dolphin.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._roi_size = config.vision.lip_roi_size
        self._event_bus = event_bus
        self._max_frames = int(LIP_BUFFER_SECONDS * LIP_BUFFER_FPS)
        # Per-face lip ROI buffer: track_id → deque of LipROI
        self._buffers: dict[int, deque[LipROI]] = defaultdict(
            lambda: deque(maxlen=self._max_frames)
        )

        # Subscribe to face lost to clean up buffers
        event_bus.subscribe(EventType.FACE_LOST, self._on_face_lost)

    def extract(self, image: np.ndarray, track: FaceTrack, timestamp: float) -> Optional[LipROI]:
        """Extract mouth ROI from an image given a face track.

        Args:
            image: BGR frame from camera
            track: FaceTrack with 468 landmarks
            timestamp: Frame timestamp

        Returns:
            LipROI with resized mouth crop, or None if extraction fails.
        """
        if track.landmarks is None or len(track.landmarks) < 96:
            return None

        h, w = image.shape[:2]

        # Get lip landmark pixel coordinates
        lip_landmarks = track.landmarks[ALL_LIP_INDICES]
        lip_x = (lip_landmarks[:, 0] * w).astype(int)
        lip_y = (lip_landmarks[:, 1] * h).astype(int)

        # Compute bounding box with padding
        x_min = max(0, lip_x.min() - 10)
        x_max = min(w, lip_x.max() + 10)
        y_min = max(0, lip_y.min() - 10)
        y_max = min(h, lip_y.max() + 10)

        if x_max <= x_min or y_max <= y_min:
            return None

        # Crop and resize
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, self._roi_size, interpolation=cv2.INTER_LINEAR)
        # Convert BGR → RGB for model input
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        roi = LipROI(image=rgb, timestamp=timestamp, track_id=track.track_id)
        self._buffers[track.track_id].append(roi)

        return roi

    def get_lip_frames(
        self,
        track_id: int,
        start_time: float,
        end_time: float,
    ) -> list[LipROI]:
        """Get buffered lip ROI frames for a face within a time window.

        Used to provide lip video sequence to Dolphin when a speech
        segment is ready for separation.
        """
        if track_id not in self._buffers:
            return []

        return [
            roi for roi in self._buffers[track_id]
            if start_time <= roi.timestamp <= end_time
        ]

    def get_recent_frames(self, track_id: int, count: int) -> list[LipROI]:
        """Get the N most recent lip frames for a face."""
        if track_id not in self._buffers:
            return []
        buf = list(self._buffers[track_id])
        return buf[-count:] if len(buf) >= count else buf

    def _on_face_lost(self, data: object) -> None:
        """Clean up lip buffer when a face is lost."""
        if isinstance(data, dict):
            track_id = data.get("track_id")
            if track_id is not None and track_id in self._buffers:
                del self._buffers[track_id]
