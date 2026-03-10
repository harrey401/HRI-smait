"""Engagement detection: gaze + face area + debounce."""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Optional

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.face_tracker import FaceTrack
from smait.perception.gaze import GazeResult

logger = logging.getLogger(__name__)

# Don't engage with faces that have rapidly changing area (walking past)
AREA_VELOCITY_THRESHOLD = 5000  # pixels²/second


class EngagementState(Enum):
    IDLE = auto()
    APPROACHING = auto()
    ENGAGED = auto()
    LOST = auto()


class EngagementDetector:
    """Determines if someone wants to interact with Jackie.

    Two signals:
    1. Distance proxy: face bounding box area (larger = closer)
    2. Gaze: L2CS-Net is_looking_at_robot

    Engagement triggered when BOTH sustained > 2 seconds (debounced).

    Rules:
    - Rapidly changing face area = walking past -> don't engage
    - Multiple faces: primary = largest face + direct gaze + DOA alignment
    - Group: wait for individual to step forward
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.engagement
        self._event_bus = event_bus
        self._state = EngagementState.IDLE
        self._target_track_id: Optional[int] = None
        self._last_doa_angle: Optional[int] = None

        # Per-face tracking state
        self._gaze_start: dict[int, float] = {}  # track_id → first gaze time
        self._area_history: dict[int, list[tuple[float, int]]] = {}  # track_id → [(time, area)]
        self._disengage_start: Optional[float] = None

        # Subscribe to DOA updates
        event_bus.subscribe(EventType.DOA_UPDATE, self._on_doa_update)

    @property
    def state(self) -> EngagementState:
        return self._state

    @property
    def target_track_id(self) -> Optional[int]:
        return self._target_track_id

    def update(
        self,
        tracks: list[FaceTrack],
        gaze_results: dict[int, GazeResult],
        timestamp: float,
    ) -> None:
        """Update engagement state based on current face tracks and gaze."""
        # Clean up stale entries
        active_ids = {t.track_id for t in tracks}
        self._gaze_start = {k: v for k, v in self._gaze_start.items() if k in active_ids}
        self._area_history = {k: v for k, v in self._area_history.items() if k in active_ids}

        if self._state == EngagementState.IDLE:
            self._check_approach(tracks, gaze_results, timestamp)
        elif self._state == EngagementState.APPROACHING:
            self._check_engage(tracks, gaze_results, timestamp)
        elif self._state == EngagementState.ENGAGED:
            self._check_disengage(tracks, gaze_results, timestamp)

    def _check_approach(
        self,
        tracks: list[FaceTrack],
        gaze_results: dict[int, GazeResult],
        timestamp: float,
    ) -> None:
        """Look for someone approaching (face + gaze toward robot)."""
        for track in tracks:
            tid = track.track_id
            gaze = gaze_results.get(tid)

            # Record area history
            if tid not in self._area_history:
                self._area_history[tid] = []
            self._area_history[tid].append((timestamp, track.face_area))
            # Keep only last 3 seconds
            self._area_history[tid] = [
                (t, a) for t, a in self._area_history[tid]
                if timestamp - t < 3.0
            ]

            # Skip faces below minimum area threshold
            if track.face_area < self._config.face_area_threshold:
                continue

            # Skip rapidly changing area (walking past)
            if self._is_walking_past(tid):
                if tid in self._gaze_start:
                    del self._gaze_start[tid]
                continue

            # Check gaze
            if gaze and gaze.is_looking_at_robot:
                if tid not in self._gaze_start:
                    self._gaze_start[tid] = timestamp
            else:
                if tid in self._gaze_start:
                    del self._gaze_start[tid]

        # Find best candidate that meets approaching criteria
        best_candidate = self._select_primary_user(tracks, gaze_results)
        if best_candidate and best_candidate.track_id in self._gaze_start:
            self._state = EngagementState.APPROACHING
            self._target_track_id = best_candidate.track_id
            logger.debug("Approaching: track_id=%d, area=%d",
                         best_candidate.track_id, best_candidate.face_area)

    def _check_engage(
        self,
        tracks: list[FaceTrack],
        gaze_results: dict[int, GazeResult],
        timestamp: float,
    ) -> None:
        """Check if approaching person should become engaged."""
        if self._target_track_id is None:
            self._state = EngagementState.IDLE
            return

        # Check target still exists
        target = None
        for t in tracks:
            if t.track_id == self._target_track_id:
                target = t
                break

        if target is None:
            self._state = EngagementState.IDLE
            self._target_track_id = None
            return

        gaze = gaze_results.get(self._target_track_id)

        # Update area history
        tid = self._target_track_id
        if tid not in self._area_history:
            self._area_history[tid] = []
        self._area_history[tid].append((timestamp, target.face_area))
        self._area_history[tid] = [
            (t, a) for t, a in self._area_history[tid]
            if timestamp - t < 3.0
        ]

        # Check gaze continuity
        if gaze and gaze.is_looking_at_robot:
            if tid not in self._gaze_start:
                self._gaze_start[tid] = timestamp
        else:
            # Gaze lost during approach
            if tid in self._gaze_start:
                del self._gaze_start[tid]
            self._state = EngagementState.IDLE
            self._target_track_id = None
            return

        # Check if sustained gaze exceeds threshold
        gaze_duration = timestamp - self._gaze_start.get(tid, timestamp)
        if gaze_duration >= self._config.min_gaze_duration_s:
            self._state = EngagementState.ENGAGED
            self._disengage_start = None
            target.is_target = True

            self._event_bus.emit(EventType.ENGAGEMENT_START, {
                "track_id": self._target_track_id,
                "face_area": target.face_area,
                "gaze_duration": gaze_duration,
                "timestamp": timestamp,
            })
            logger.info("Engaged: track_id=%d after %.1fs gaze",
                        self._target_track_id, gaze_duration)

    def _check_disengage(
        self,
        tracks: list[FaceTrack],
        gaze_results: dict[int, GazeResult],
        timestamp: float,
    ) -> None:
        """Check if engaged person is disengaging."""
        if self._target_track_id is None:
            return

        # Check target still exists
        target = None
        for t in tracks:
            if t.track_id == self._target_track_id:
                target = t
                break

        if target is None:
            # Face lost — don't immediately disengage (grace period handled by SessionManager)
            return

        gaze = gaze_results.get(self._target_track_id)
        looking = gaze and gaze.is_looking_at_robot if gaze else False

        if not looking:
            if self._disengage_start is None:
                self._disengage_start = timestamp
            elif timestamp - self._disengage_start >= self._config.disengage_gaze_timeout_s:
                self._state = EngagementState.IDLE
                old_target = self._target_track_id
                self._target_track_id = None
                self._disengage_start = None
                target.is_target = False

                self._event_bus.emit(EventType.ENGAGEMENT_LOST, {
                    "track_id": old_target,
                    "reason": "gaze_timeout",
                    "timestamp": timestamp,
                })
                logger.info("Engagement lost: track_id=%d (gaze timeout)", old_target)
        else:
            self._disengage_start = None

    def _doa_score_for_face(
        self,
        track: FaceTrack,
        frame_width: int = 640,
        camera_fov_deg: float = 60.0,
    ) -> float:
        """Return DOA alignment multiplier based on angular proximity.

        Maps the face's pixel position to a camera angle, then computes how
        closely it aligns with the DOA direction. Returns 1.0 for perfect
        alignment and 0.5 for a face 90+ degrees away from DOA.

        Args:
            track: FaceTrack with bbox=(x, y, w, h)
            frame_width: Camera frame width in pixels (default 640)
            camera_fov_deg: Camera horizontal field of view in degrees (default 60)

        Returns:
            Multiplier in [0.5, 1.0] — higher means closer to DOA direction.
        """
        if self._last_doa_angle is None:
            return 1.0  # No DOA data — no penalty applied

        # FaceTrack.bbox is (x, y, w, h)
        x, _y, w, _h = track.bbox
        face_center_x = x + w / 2

        # Map pixel X to camera angle: center=0, left=negative, right=positive
        normalized = (face_center_x / frame_width) - 0.5  # [-0.5, 0.5]
        face_angle_deg = normalized * camera_fov_deg

        # DOA: 0=front, negative=left, positive=right
        angular_distance = abs(face_angle_deg - self._last_doa_angle)
        return max(0.5, 1.0 - angular_distance / 90.0)

    def _select_primary_user(
        self,
        tracks: list[FaceTrack],
        gaze_results: dict[int, GazeResult],
    ) -> Optional[FaceTrack]:
        """Select the primary user: largest face + direct gaze + DOA alignment."""
        candidates = []
        for track in tracks:
            gaze = gaze_results.get(track.track_id)
            if gaze and gaze.is_looking_at_robot:
                if track.face_area >= self._config.face_area_threshold:
                    if not self._is_walking_past(track.track_id):
                        score = track.face_area
                        # Per-face DOA alignment scoring (angular proximity)
                        score *= self._doa_score_for_face(track)
                        candidates.append((score, track))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _is_walking_past(self, track_id: int) -> bool:
        """Detect rapidly changing face area (person walking past)."""
        history = self._area_history.get(track_id, [])
        if len(history) < 3:
            return False

        # Compute area velocity over last second
        recent = [(t, a) for t, a in history if history[-1][0] - t < 1.0]
        if len(recent) < 2:
            return False

        dt = recent[-1][0] - recent[0][0]
        if dt < 0.1:
            return False

        da = abs(recent[-1][1] - recent[0][1])
        velocity = da / dt

        return velocity > AREA_VELOCITY_THRESHOLD

    def _on_doa_update(self, data: object) -> None:
        if isinstance(data, dict):
            self._last_doa_angle = data.get("angle")

    def reset(self) -> None:
        """Reset engagement state (e.g., on session end)."""
        self._state = EngagementState.IDLE
        self._target_track_id = None
        self._gaze_start.clear()
        self._area_history.clear()
        self._disengage_start = None
