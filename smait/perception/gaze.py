"""L2CS-Net gaze estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.face_tracker import FaceTrack

logger = logging.getLogger(__name__)


@dataclass
class GazeResult:
    """Per-face gaze estimation result."""
    track_id: int
    yaw_deg: float
    pitch_deg: float
    is_looking_at_robot: bool
    timestamp: float


class GazeEstimator:
    """L2CS-Net: lightweight gaze estimation (~0.3GB VRAM).

    Input: face crop from FaceTracker bbox
    Output: GazeResult(yaw_deg, pitch_deg, is_looking_at_robot)

    is_looking_at_robot = |yaw| < 30 deg AND |pitch| < 20 deg
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._yaw_threshold = config.gaze.yaw_threshold
        self._pitch_threshold = config.gaze.pitch_threshold
        self._event_bus = event_bus
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform_size = (224, 224)

    async def init_model(self) -> None:
        """Load L2CS-Net model.

        L2CS-Net uses a ResNet50 backbone with two separate FC heads
        for yaw and pitch prediction (binned classification + regression).
        """
        logger.info("Loading L2CS-Net gaze model...")
        try:
            from l2cs import Pipeline as L2CSPipeline
            self._l2cs_pipeline = L2CSPipeline(
                weights=None,  # Auto-download
                arch='ResNet50',
                device=self._device,
            )
            logger.info("L2CS-Net loaded on %s", self._device)
        except ImportError:
            logger.warning(
                "L2CS-Net not installed. Gaze estimation will use head pose fallback. "
                "Install from: pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main"
            )
            self._l2cs_pipeline = None
        except Exception as exc:
            logger.warning(
                "L2CS-Net failed to load (weights download issue?): %s. "
                "Using head pose fallback.", exc
            )
            self._l2cs_pipeline = None

    def estimate(
        self,
        image: np.ndarray,
        track: FaceTrack,
        timestamp: float,
    ) -> GazeResult:
        """Estimate gaze direction for a tracked face.

        Falls back to head pose if L2CS-Net is not available.
        """
        if self._l2cs_pipeline is not None:
            return self._estimate_l2cs(image, track, timestamp)
        return self._estimate_from_head_pose(track, timestamp)

    def _estimate_l2cs(
        self,
        image: np.ndarray,
        track: FaceTrack,
        timestamp: float,
    ) -> GazeResult:
        """Estimate gaze using L2CS-Net pipeline."""
        x, y, w, h = track.bbox
        # Pad the face crop slightly
        pad = int(max(w, h) * 0.1)
        img_h, img_w = image.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return self._estimate_from_head_pose(track, timestamp)

        try:
            results = self._l2cs_pipeline.step(face_crop)
            if results and hasattr(results, "yaw") and len(results.yaw) > 0:
                yaw = float(results.yaw[0])
                pitch = float(results.pitch[0])
            else:
                return self._estimate_from_head_pose(track, timestamp)
        except Exception:
            logger.debug("L2CS-Net inference failed, using head pose fallback")
            return self._estimate_from_head_pose(track, timestamp)

        is_looking = abs(yaw) < self._yaw_threshold and abs(pitch) < self._pitch_threshold

        result = GazeResult(
            track_id=track.track_id,
            yaw_deg=yaw,
            pitch_deg=pitch,
            is_looking_at_robot=is_looking,
            timestamp=timestamp,
        )

        self._event_bus.emit(EventType.GAZE_UPDATE, result)
        return result

    def _estimate_from_head_pose(self, track: FaceTrack, timestamp: float) -> GazeResult:
        """Fallback: use head pose from face tracker as gaze proxy."""
        yaw = track.head_yaw
        pitch = track.head_pitch
        is_looking = abs(yaw) < self._yaw_threshold and abs(pitch) < self._pitch_threshold

        result = GazeResult(
            track_id=track.track_id,
            yaw_deg=yaw,
            pitch_deg=pitch,
            is_looking_at_robot=is_looking,
            timestamp=timestamp,
        )

        self._event_bus.emit(EventType.GAZE_UPDATE, result)
        return result
