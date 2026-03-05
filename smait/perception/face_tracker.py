"""MediaPipe Face Mesh with persistent track IDs via IOU matching."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

# IOU threshold for re-associating faces across frames
IOU_THRESHOLD = 0.3
# Time before a lost face is removed from tracking
FACE_LOST_TIMEOUT_S = 2.0


@dataclass
class FaceTrack:
    """A persistently tracked face."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray             # (468, 3) normalized landmarks
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.0
    is_target: bool = False
    face_area: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


class FaceTracker:
    """MediaPipe Face Mesh: 468 landmarks per face, persistent track IDs.

    - Re-association after brief occlusion via IOU + centroid matching
    - Emits FACE_DETECTED, FACE_LOST, FACE_UPDATED events
    - Max faces configured via VisionConfig.max_faces
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.vision
        self._event_bus = event_bus
        self._next_id = 1
        self._tracks: dict[int, FaceTrack] = {}

        # MediaPipe Face Mesh
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self._config.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self._config.min_face_confidence,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, image: np.ndarray, timestamp: float) -> list[FaceTrack]:
        """Process a BGR frame and return updated face tracks."""
        h, w = image.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        current_detections: list[tuple[tuple[int, int, int, int], np.ndarray, float]] = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
                    dtype=np.float32,
                )

                # Compute bounding box from landmarks
                xs = (landmarks[:, 0] * w).astype(int)
                ys = (landmarks[:, 1] * h).astype(int)
                x_min, x_max = max(0, xs.min()), min(w, xs.max())
                y_min, y_max = max(0, ys.min()), min(h, ys.max())
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                area = bbox[2] * bbox[3]

                # Estimate head pose from key landmarks (nose tip, chin, etc.)
                yaw, pitch = self._estimate_head_pose(landmarks, w, h)

                # Confidence from face area (proxy — MediaPipe doesn't expose per-face confidence directly)
                confidence = min(1.0, area / 20000.0)

                current_detections.append((bbox, landmarks, confidence))

        # Match detections to existing tracks
        updated_tracks = self._match_and_update(current_detections, timestamp)

        # Check for lost faces
        lost_ids = []
        for track_id, track in self._tracks.items():
            if track_id not in {t.track_id for t in updated_tracks}:
                if timestamp - track.last_seen > FACE_LOST_TIMEOUT_S:
                    lost_ids.append(track_id)

        for track_id in lost_ids:
            track = self._tracks.pop(track_id)
            self._event_bus.emit(EventType.FACE_LOST, {
                "track_id": track_id,
                "last_bbox": track.bbox,
                "timestamp": timestamp,
            })
            logger.debug("Face lost: track_id=%d", track_id)

        return updated_tracks

    def _match_and_update(
        self,
        detections: list[tuple[tuple[int, int, int, int], np.ndarray, float]],
        timestamp: float,
    ) -> list[FaceTrack]:
        """Match current detections to existing tracks via IOU."""
        if not detections:
            return []

        updated = []
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # Compute IOU between all tracks and detections
        for det_idx, (det_bbox, det_landmarks, det_conf) in enumerate(detections):
            best_iou = 0.0
            best_track_id = None

            for track_id, track in self._tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._compute_iou(track.bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= IOU_THRESHOLD and best_track_id is not None:
                # Update existing track
                track = self._tracks[best_track_id]
                track.bbox = det_bbox
                track.landmarks = det_landmarks
                track.last_seen = timestamp
                track.confidence = det_conf
                track.face_area = det_bbox[2] * det_bbox[3]
                yaw, pitch = self._estimate_head_pose(det_landmarks,
                                                       det_bbox[2] + det_bbox[0],
                                                       det_bbox[3] + det_bbox[1])
                track.head_yaw = yaw
                track.head_pitch = pitch

                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)
                updated.append(track)

                self._event_bus.emit(EventType.FACE_UPDATED, {
                    "track": track,
                    "timestamp": timestamp,
                })

        # Create new tracks for unmatched detections
        for det_idx, (det_bbox, det_landmarks, det_conf) in enumerate(detections):
            if det_idx in matched_det_indices:
                continue

            track_id = self._next_id
            self._next_id += 1
            area = det_bbox[2] * det_bbox[3]
            yaw, pitch = self._estimate_head_pose(det_landmarks,
                                                   det_bbox[2] + det_bbox[0],
                                                   det_bbox[3] + det_bbox[1])

            track = FaceTrack(
                track_id=track_id,
                bbox=det_bbox,
                landmarks=det_landmarks,
                head_yaw=yaw,
                head_pitch=pitch,
                last_seen=timestamp,
                confidence=det_conf,
                face_area=area,
            )
            self._tracks[track_id] = track
            updated.append(track)

            self._event_bus.emit(EventType.FACE_DETECTED, {
                "track": track,
                "timestamp": timestamp,
            })
            logger.debug("New face detected: track_id=%d, area=%d", track_id, area)

        return updated

    @staticmethod
    def _compute_iou(
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _estimate_head_pose(landmarks: np.ndarray, w: int, h: int) -> tuple[float, float]:
        """Rough head pose estimation from key MediaPipe landmarks.

        Uses nose tip (1), chin (152), left eye outer (33), right eye outer (263).
        Returns (yaw_deg, pitch_deg).
        """
        if len(landmarks) < 468:
            return 0.0, 0.0

        nose = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        # Yaw: difference in x between nose and midpoint of eyes
        eye_mid_x = (left_eye[0] + right_eye[0]) / 2
        yaw = (nose[0] - eye_mid_x) * 180.0  # Rough degrees

        # Pitch: nose-chin vertical angle
        dy = chin[1] - nose[1]
        dz = chin[2] - nose[2]
        pitch = np.degrees(np.arctan2(dz, dy)) if dy != 0 else 0.0

        return float(yaw), float(pitch)

    @property
    def tracks(self) -> dict[int, FaceTrack]:
        return self._tracks

    def get_target_face(self) -> Optional[FaceTrack]:
        """Get the current target face (the one marked is_target)."""
        for track in self._tracks.values():
            if track.is_target:
                return track
        return None

    def set_target(self, track_id: int) -> None:
        """Set a face as the interaction target."""
        for tid, track in self._tracks.items():
            track.is_target = (tid == track_id)

    def close(self) -> None:
        self._face_mesh.close()
