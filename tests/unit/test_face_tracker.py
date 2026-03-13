"""Unit tests for FaceTracker IOU matching and event emission.

Tests cover:
- _compute_iou() correctness for identical, non-overlapping, and partial bboxes
- IOU symmetry property
- _estimate_head_pose() with valid and insufficient landmarks
- _match_and_update() creates new tracks and reuses existing track IDs
- FACE_DETECTED and FACE_LOST events
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smait.core.events import EventBus, EventType
from smait.perception.face_tracker import FaceTrack, FaceTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks(n: int = 478) -> np.ndarray:
    """Create synthetic normalized landmarks array of shape (n, 3)."""
    rng = np.random.default_rng(42)
    lm = rng.uniform(0.3, 0.7, size=(n, 3)).astype(np.float32)
    return lm


def _make_face_detection(
    bbox: tuple[int, int, int, int] = (10, 10, 80, 80),
    n_landmarks: int = 478,
    confidence: float = 0.9,
) -> tuple[tuple[int, int, int, int], np.ndarray, float]:
    """Create a synthetic (bbox, landmarks, confidence) detection tuple."""
    return (bbox, _make_landmarks(n_landmarks), confidence)


def _make_tracker_with_mock(config, event_bus) -> FaceTracker:
    """Create a FaceTracker with MediaPipe FaceLandmarker mocked out.

    Patches the vision module to avoid loading the real model file.
    """
    mock_landmarker = MagicMock()
    with patch("smait.perception.face_tracker.vision") as mock_vision, \
         patch("smait.perception.face_tracker._ensure_model", return_value="/fake/model.task"):
        mock_vision.FaceLandmarkerOptions = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = mock_landmarker
        mock_vision.RunningMode.VIDEO = "VIDEO"
        tracker = FaceTracker(config, event_bus)
    return tracker


# ---------------------------------------------------------------------------
# IOU Tests
# ---------------------------------------------------------------------------

def test_iou_identical_bboxes():
    """_compute_iou(bbox, bbox) == 1.0 for identical bounding boxes."""
    bbox = (10, 10, 80, 80)
    result = FaceTracker._compute_iou(bbox, bbox)
    assert result == pytest.approx(1.0)


def test_iou_no_overlap():
    """_compute_iou returns 0.0 for non-overlapping bounding boxes."""
    bbox1 = (0, 0, 50, 50)
    bbox2 = (100, 100, 50, 50)
    result = FaceTracker._compute_iou(bbox1, bbox2)
    assert result == pytest.approx(0.0)


def test_iou_partial_overlap():
    """_compute_iou returns value between 0 and 1 for partial overlap."""
    bbox1 = (0, 0, 100, 100)
    bbox2 = (50, 50, 100, 100)
    result = FaceTracker._compute_iou(bbox1, bbox2)
    assert 0.0 < result < 1.0


def test_iou_symmetric():
    """_compute_iou(a, b) == _compute_iou(b, a): IOU is symmetric."""
    bbox_a = (0, 0, 100, 100)
    bbox_b = (60, 60, 80, 80)
    assert FaceTracker._compute_iou(bbox_a, bbox_b) == pytest.approx(
        FaceTracker._compute_iou(bbox_b, bbox_a)
    )


# ---------------------------------------------------------------------------
# Head Pose Estimation Tests
# ---------------------------------------------------------------------------

def test_head_pose_estimation():
    """_estimate_head_pose returns (float, float) from valid 478-point landmarks."""
    landmarks = _make_landmarks(478)
    yaw, pitch = FaceTracker._estimate_head_pose(landmarks, w=640, h=480)
    assert isinstance(yaw, float)
    assert isinstance(pitch, float)


def test_head_pose_short_landmarks():
    """_estimate_head_pose returns (0.0, 0.0) when landmarks has < 468 points."""
    landmarks = _make_landmarks(100)
    yaw, pitch = FaceTracker._estimate_head_pose(landmarks, w=640, h=480)
    assert yaw == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _match_and_update Tests
# ---------------------------------------------------------------------------

def test_match_and_update_creates_new_track(config, event_bus):
    """_match_and_update with no existing tracks creates a new FaceTrack with track_id=1."""
    tracker = _make_tracker_with_mock(config, event_bus)

    detection = _make_face_detection(bbox=(10, 10, 80, 80))
    result = tracker._match_and_update([detection], timestamp=1.0)

    assert len(result) == 1
    assert result[0].track_id == 1
    assert result[0].bbox == (10, 10, 80, 80)
    assert result[0].face_area == 80 * 80


def test_match_and_update_reuses_track_id(config, event_bus):
    """_match_and_update reuses existing track_id when IOU >= 0.3."""
    tracker = _make_tracker_with_mock(config, event_bus)

    # First detection — creates track_id=1
    det1 = _make_face_detection(bbox=(10, 10, 80, 80))
    first = tracker._match_and_update([det1], timestamp=1.0)
    assert first[0].track_id == 1

    # Second detection — slightly shifted but high IOU (should reuse track_id=1)
    det2 = _make_face_detection(bbox=(12, 12, 80, 80))
    second = tracker._match_and_update([det2], timestamp=1.1)

    assert len(second) == 1
    assert second[0].track_id == 1, (
        "Same track should be reused for high-IOU detection"
    )


def test_match_and_update_emits_face_detected(config, event_bus):
    """FACE_DETECTED event is emitted when a new track is created."""
    tracker = _make_tracker_with_mock(config, event_bus)

    detected_events: list = []
    event_bus.subscribe(EventType.FACE_DETECTED, lambda data: detected_events.append(data))

    det = _make_face_detection(bbox=(20, 20, 60, 60))
    tracker._match_and_update([det], timestamp=0.5)

    assert len(detected_events) == 1
    evt = detected_events[0]
    assert "track" in evt
    assert evt["track"].track_id == 1


# ---------------------------------------------------------------------------
# FACE_LOST Event Test
# ---------------------------------------------------------------------------

def _make_mock_normalized_landmark(x: float, y: float, z: float = 0.0) -> MagicMock:
    """Create a mock NormalizedLandmark with x, y, z attributes."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_mock_face_landmarks_list(n: int = 478) -> list:
    """Create a list of mock NormalizedLandmark objects (new API format)."""
    landmarks = []
    for i in range(n):
        x = 0.3 + (i % 20) * 0.02  # x range 0.3 - 0.68
        y = 0.3 + (i // 20) * 0.02  # y range 0.3 - 0.74
        landmarks.append(_make_mock_normalized_landmark(x, y, 0.0))
    return landmarks


def test_face_lost_emitted_after_timeout(config, event_bus):
    """FACE_LOST event emitted when a track is not seen for > 2.0s."""
    lost_events: list = []
    event_bus.subscribe(EventType.FACE_LOST, lambda data: lost_events.append(data))

    mock_landmarker = MagicMock()

    with patch("smait.perception.face_tracker.vision") as mock_vision, \
         patch("smait.perception.face_tracker._ensure_model", return_value="/fake/model.task"), \
         patch("smait.perception.face_tracker.mp") as mock_mp:
        mock_vision.FaceLandmarkerOptions = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = mock_landmarker
        mock_vision.RunningMode.VIDEO = "VIDEO"

        tracker = FaceTracker(config, event_bus)

        # Frame 1: One face detected at t=0.0
        mock_results_with_face = MagicMock()
        mock_results_with_face.face_landmarks = [_make_mock_face_landmarks_list(478)]
        mock_landmarker.detect_for_video.return_value = mock_results_with_face

        mock_mp.Image.return_value = MagicMock()
        mock_mp.ImageFormat.SRGB = "SRGB"

        with patch("smait.perception.face_tracker.cv2") as mock_cv2:
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            tracks = tracker.process_frame(frame, timestamp=0.0)
            assert len(tracks) == 1, "One face should be detected"

            # Frame 2: No faces at t=2.1s (> FACE_LOST_TIMEOUT_S=2.0)
            mock_results_no_face = MagicMock()
            mock_results_no_face.face_landmarks = None
            mock_landmarker.detect_for_video.return_value = mock_results_no_face

            tracker.process_frame(frame, timestamp=2.1)

    assert len(lost_events) == 1, "FACE_LOST must fire after 2.0s timeout"
    evt = lost_events[0]
    assert "track_id" in evt
    assert "last_bbox" in evt
    assert "timestamp" in evt
    assert evt["track_id"] == 1
    assert evt["timestamp"] == pytest.approx(2.1)
