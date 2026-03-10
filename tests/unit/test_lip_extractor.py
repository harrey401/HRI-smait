"""Tests for LipExtractor — ROI shape, buffer operations, and FACE_LOST cleanup.

TDD Plan 03-01: RED phase verifies default config bug (96x96), GREEN fixes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.face_tracker import FaceTrack
from smait.perception.lip_extractor import LipExtractor, LipROI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face_track(
    track_id: int = 1,
    landmarks_shape: tuple = (468, 3),
) -> FaceTrack:
    """Create a FaceTrack with lip landmarks set to normalized coords ~(0.5, 0.6).

    MediaPipe lip indices 61-68 (outer) and 78-95 (inner) are set to
    normalized x=0.5, y=0.6 so they fall inside a 480x640 frame (pixel
    coords ~320, 288 — well within bounds).
    """
    if landmarks_shape[0] >= 96:
        landmarks = np.zeros(landmarks_shape, dtype=np.float32)
        # Outer lip indices 61-68
        for i in range(61, 69):
            landmarks[i] = [0.5, 0.6, 0.0]
        # Inner lip indices 78-95
        for i in range(78, 96):
            landmarks[i] = [0.5, 0.6, 0.0]
    else:
        landmarks = np.zeros(landmarks_shape, dtype=np.float32)

    return FaceTrack(
        track_id=track_id,
        bbox=(280, 250, 80, 60),
        landmarks=landmarks,
        head_yaw=0.0,
        head_pitch=0.0,
        last_seen=0.0,
        confidence=0.9,
        is_target=False,
        face_area=4800,
    )


def _make_frame() -> np.ndarray:
    """Create a synthetic 480x640 BGR frame with a bright mouth area."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # White region at the mouth area so crop is non-trivial
    frame[250:310, 280:360] = 200
    return frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor(config, event_bus):
    """Fresh LipExtractor with default config."""
    return LipExtractor(config, event_bus)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lip_roi_output_shape(extractor):
    """LipExtractor.extract() returns LipROI with image.shape == (88, 88, 3)."""
    frame = _make_frame()
    track = _make_face_track(track_id=1)

    roi = extractor.extract(frame, track, timestamp=1.0)

    assert roi is not None, "extract() returned None unexpectedly"
    assert roi.image.shape == (88, 88, 3), (
        f"Expected (88, 88, 3), got {roi.image.shape}. "
        "Check VisionConfig.lip_roi_size default."
    )


def test_lip_roi_output_dtype(extractor):
    """LipROI.image.dtype == np.uint8."""
    frame = _make_frame()
    track = _make_face_track(track_id=1)

    roi = extractor.extract(frame, track, timestamp=1.0)

    assert roi is not None
    assert roi.image.dtype == np.uint8, (
        f"Expected uint8, got {roi.image.dtype}"
    )


def test_extract_returns_none_for_insufficient_landmarks(extractor):
    """extract() returns None when track.landmarks has fewer than 96 points."""
    frame = _make_frame()
    track = _make_face_track(track_id=2, landmarks_shape=(50, 3))

    roi = extractor.extract(frame, track, timestamp=1.0)

    assert roi is None, "Should return None for fewer than 96 landmarks"


def test_extract_returns_none_for_none_landmarks(extractor):
    """extract() returns None when track.landmarks is None."""
    frame = _make_frame()
    track = _make_face_track(track_id=3)
    track.landmarks = None  # Override to None

    roi = extractor.extract(frame, track, timestamp=1.0)

    assert roi is None, "Should return None when landmarks is None"


def test_get_lip_frames_time_filter(extractor):
    """get_lip_frames() returns only frames within [start_time, end_time] window."""
    frame = _make_frame()
    track = _make_face_track(track_id=4)

    # Extract frames at different timestamps
    extractor.extract(frame, track, timestamp=1.0)
    extractor.extract(frame, track, timestamp=3.0)
    extractor.extract(frame, track, timestamp=5.0)
    extractor.extract(frame, track, timestamp=7.0)

    frames = extractor.get_lip_frames(track_id=4, start_time=2.0, end_time=6.0)

    assert len(frames) == 2, f"Expected 2 frames in [2.0, 6.0], got {len(frames)}"
    timestamps = [f.timestamp for f in frames]
    assert 3.0 in timestamps
    assert 5.0 in timestamps


def test_get_lip_frames_empty_for_unknown_track(extractor):
    """get_lip_frames() returns [] for a track_id not in buffer."""
    frames = extractor.get_lip_frames(track_id=999, start_time=0.0, end_time=10.0)
    assert frames == [], f"Expected [], got {frames}"


def test_get_recent_frames(extractor):
    """get_recent_frames(track_id, 3) returns last 3 frames from buffer."""
    frame = _make_frame()
    track = _make_face_track(track_id=5)

    # Extract 5 frames
    for i in range(5):
        extractor.extract(frame, track, timestamp=float(i))

    recent = extractor.get_recent_frames(track_id=5, count=3)

    assert len(recent) == 3, f"Expected 3 recent frames, got {len(recent)}"
    # Should be the last 3 (timestamps 2.0, 3.0, 4.0)
    timestamps = [f.timestamp for f in recent]
    assert timestamps == [2.0, 3.0, 4.0], f"Expected [2.0, 3.0, 4.0], got {timestamps}"


def test_face_lost_clears_buffer(extractor, event_bus):
    """Emitting FACE_LOST event with {"track_id": N} removes that track's buffer."""
    frame = _make_frame()
    track = _make_face_track(track_id=6)

    # Extract a frame to populate the buffer
    roi = extractor.extract(frame, track, timestamp=1.0)
    assert roi is not None

    # Verify buffer has frames
    frames_before = extractor.get_lip_frames(track_id=6, start_time=0.0, end_time=10.0)
    assert len(frames_before) == 1, "Buffer should have 1 frame before FACE_LOST"

    # Emit FACE_LOST event
    event_bus.emit(EventType.FACE_LOST, {"track_id": 6})

    # Buffer should now be empty for this track
    frames_after = extractor.get_lip_frames(track_id=6, start_time=0.0, end_time=10.0)
    assert frames_after == [], f"Buffer should be empty after FACE_LOST, got {frames_after}"


def test_buffer_appends_on_extract(extractor):
    """Each successful extract() call appends to the per-track buffer."""
    frame = _make_frame()
    track = _make_face_track(track_id=7)

    for i in range(4):
        roi = extractor.extract(frame, track, timestamp=float(i))
        assert roi is not None, f"extract() returned None at timestamp {float(i)}"

    all_frames = extractor.get_lip_frames(track_id=7, start_time=0.0, end_time=10.0)
    assert len(all_frames) == 4, f"Expected 4 frames in buffer, got {len(all_frames)}"
