"""Unit tests for EngagementDetector state machine.

Tests cover:
- IDLE -> APPROACHING -> ENGAGED transitions after 2s sustained gaze
- Gaze break resets to IDLE during APPROACHING
- ENGAGEMENT_START and ENGAGEMENT_LOST events
- Walking-past filter suppresses rapidly-moving faces
- DOA bonus scoring in primary user selection
- reset() clears all state
"""

from __future__ import annotations

import numpy as np
import pytest

from smait.core.events import EventBus, EventType
from smait.perception.engagement import (
    EngagementDetector,
    EngagementState,
)
from smait.perception.face_tracker import FaceTrack
from smait.perception.gaze import GazeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(track_id: int = 1, face_area: int = 10000) -> FaceTrack:
    """Create a FaceTrack with constant face_area and dummy landmarks."""
    landmarks = np.random.rand(468, 3).astype(np.float32)
    return FaceTrack(
        track_id=track_id,
        bbox=(10, 10, 100, 100),
        landmarks=landmarks,
        head_yaw=5.0,
        head_pitch=3.0,
        last_seen=0.0,
        confidence=0.9,
        is_target=False,
        face_area=face_area,
    )


def _make_gaze(track_id: int, looking: bool, t: float) -> GazeResult:
    """Create a GazeResult for a given track_id and gaze state."""
    return GazeResult(
        track_id=track_id,
        yaw_deg=5.0 if looking else 90.0,
        pitch_deg=3.0,
        is_looking_at_robot=looking,
        timestamp=t,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector(config, event_bus):
    """EngagementDetector with default config."""
    return EngagementDetector(config, event_bus)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_state_is_idle(detector):
    """EngagementDetector starts in IDLE state with no target."""
    assert detector.state == EngagementState.IDLE
    assert detector.target_track_id is None


def test_sustained_gaze_reaches_engaged(detector):
    """IDLE -> APPROACHING -> ENGAGED after 2.1s of sustained gaze.

    face_area=10000 (above 3000 threshold) and is_looking_at_robot=True.
    """
    track = _make_track(track_id=1, face_area=10000)
    t = 0.0
    while t <= 2.1:
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)
        t = round(t + 0.1, 1)

    assert detector.state == EngagementState.ENGAGED
    assert detector.target_track_id == 1


def test_gaze_break_resets_to_idle(detector):
    """Gaze loss during APPROACHING resets state to IDLE."""
    track = _make_track(track_id=1, face_area=10000)

    # Drive to APPROACHING (5 frames of gaze = 0.5s, not yet ENGAGED)
    for i in range(5):
        t = round(i * 0.1, 1)
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)

    # Verify we are APPROACHING (not yet ENGAGED)
    assert detector.state == EngagementState.APPROACHING

    # Break gaze — should reset to IDLE
    gaze_off = _make_gaze(1, False, 0.5)
    detector.update([track], {1: gaze_off}, 0.5)

    assert detector.state == EngagementState.IDLE


def test_face_below_area_threshold_ignored(detector):
    """Face with face_area < 3000 does not trigger APPROACHING."""
    # face_area=1000 is well below the 3000 threshold
    track = _make_track(track_id=1, face_area=1000)

    for i in range(25):
        t = round(i * 0.1, 1)
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)

    # State should remain IDLE since face is too small
    assert detector.state == EngagementState.IDLE


def test_engagement_start_event_emitted(event_bus, config):
    """ENGAGEMENT_START event emitted with correct data when reaching ENGAGED."""
    detector = EngagementDetector(config, event_bus)
    events_received: list = []
    event_bus.subscribe(EventType.ENGAGEMENT_START, lambda data: events_received.append(data))

    track = _make_track(track_id=1, face_area=10000)
    t = 0.0
    while t <= 2.1:
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)
        t = round(t + 0.1, 1)

    assert len(events_received) == 1, "ENGAGEMENT_START must fire exactly once"
    evt = events_received[0]
    assert "track_id" in evt
    assert "face_area" in evt
    assert "gaze_duration" in evt
    assert "timestamp" in evt
    assert evt["track_id"] == 1
    assert evt["face_area"] == 10000


def test_disengage_after_gaze_timeout(detector):
    """ENGAGED -> IDLE after disengage_gaze_timeout_s (3.0s) without gaze."""
    track = _make_track(track_id=1, face_area=10000)

    # Drive to ENGAGED
    t = 0.0
    while t <= 2.1:
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)
        t = round(t + 0.1, 1)

    assert detector.state == EngagementState.ENGAGED

    # Now hold gaze off for > 3.0s (disengage_gaze_timeout_s)
    base_t = t
    for i in range(32):  # 3.1s in 0.1s steps
        t = round(base_t + i * 0.1, 1)
        gaze = _make_gaze(1, False, t)
        detector.update([track], {1: gaze}, t)

    assert detector.state == EngagementState.IDLE


def test_engagement_lost_event_emitted(event_bus, config):
    """ENGAGEMENT_LOST event emitted with reason='gaze_timeout' on disengage."""
    detector = EngagementDetector(config, event_bus)
    lost_events: list = []
    event_bus.subscribe(EventType.ENGAGEMENT_LOST, lambda data: lost_events.append(data))

    track = _make_track(track_id=1, face_area=10000)

    # Drive to ENGAGED
    t = 0.0
    while t <= 2.1:
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)
        t = round(t + 0.1, 1)

    assert detector.state == EngagementState.ENGAGED

    # Hold gaze off > 3.0s
    base_t = t
    for i in range(32):
        t = round(base_t + i * 0.1, 1)
        gaze = _make_gaze(1, False, t)
        detector.update([track], {1: gaze}, t)

    assert len(lost_events) == 1
    assert lost_events[0]["reason"] == "gaze_timeout"
    assert lost_events[0]["track_id"] == 1


def test_walking_past_not_engaged(detector):
    """Face with rapidly changing face_area (>5000 px^2/s velocity) stays IDLE.

    Simulates a person walking past: face_area increases from 10000 to 20000
    over 1 second (10000 px^2/s velocity >> 5000 threshold).

    The walking-past filter requires area history to be built before the filter
    can trigger. We prime history with gaze OFF (so the state doesn't transition
    to APPROACHING during the priming phase), then enable gaze while area
    velocity remains high.
    """
    # Phase 1: Prime area history with gaze OFF and rapidly increasing face_area
    # Area goes +1000/step, so velocity ~ 10000 px^2/s >> 5000 threshold.
    # Without gaze, state stays IDLE and area history accumulates.
    for i in range(5):
        t = round(i * 0.1, 1)
        face_area = 10000 + i * 1000
        track = _make_track(track_id=1, face_area=face_area)
        gaze = _make_gaze(1, False, t)
        detector.update([track], {1: gaze}, t)

    # After 5 frames, walking-past should be detected (velocity > 5000)
    assert detector._is_walking_past(1), (
        "Expected is_walking_past=True after rapid area growth"
    )

    # Phase 2: Enable gaze — walking-past filter should now suppress APPROACHING
    for i in range(5, 11):
        t = round(i * 0.1, 1)
        face_area = 10000 + i * 1000
        track = _make_track(track_id=1, face_area=face_area)
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)

    assert detector.state == EngagementState.IDLE, (
        f"Expected IDLE (walking-past suppressed), got {detector.state}"
    )


def test_doa_bonus_scoring(event_bus, config):
    """DOA_UPDATE sets _last_doa_angle; primary user selection gives 1.2x bonus."""
    detector = EngagementDetector(config, event_bus)

    # Emit DOA_UPDATE to set _last_doa_angle
    event_bus.emit(EventType.DOA_UPDATE, {"angle": 45})
    assert detector._last_doa_angle == 45

    # Create two tracks: track 2 has a larger face_area, track 1 is smaller
    # With DOA bonus active, both get 1.2x, but the larger wins
    track1 = _make_track(track_id=1, face_area=5000)
    track2 = _make_track(track_id=2, face_area=8000)

    gaze1 = _make_gaze(1, True, 0.0)
    gaze2 = _make_gaze(2, True, 0.0)

    # Call _select_primary_user directly to verify selection logic
    # Both candidates are valid: above threshold, looking, not walking past
    result = detector._select_primary_user(
        [track1, track2],
        {1: gaze1, 2: gaze2},
    )

    # Track 2 should win (larger face_area * 1.2 > track1's face_area * 1.2)
    assert result is not None
    assert result.track_id == 2


def test_reset_clears_state(detector):
    """reset() returns detector to IDLE with no target."""
    track = _make_track(track_id=1, face_area=10000)

    # Drive to ENGAGED
    t = 0.0
    while t <= 2.1:
        gaze = _make_gaze(1, True, t)
        detector.update([track], {1: gaze}, t)
        t = round(t + 0.1, 1)

    assert detector.state == EngagementState.ENGAGED

    # Reset and verify clean state
    detector.reset()

    assert detector.state == EngagementState.IDLE
    assert detector.target_track_id is None


def test_doa_angle_disambiguates_multiple_faces(event_bus, config):
    """DOA angle selects face closest in angular position, not just largest face area.

    Two faces with equal area:
    - Face A: left of frame (bbox x=50, w=120 → center_x=110), face_area=10000
      → normalized = (110/640) - 0.5 = -0.328, face_angle = -19.7°, distance to DOA=0 is 19.7°
      → DOA multiplier = max(0.5, 1 - 19.7/90) ≈ 0.781
      → final score = 10000 * 0.781 = 7810

    - Face B: near center (bbox x=300, w=90 → center_x=345), face_area=10000
      → normalized = (345/640) - 0.5 = 0.039, face_angle = 2.34°, distance to DOA=0 is 2.34°
      → DOA multiplier = max(0.5, 1 - 2.34/90) ≈ 0.974
      → final score = 10000 * 0.974 = 9740

    DOA angle=0 (front/center). Face B is angularly closer to DOA=0.
    Per-face DOA scoring selects face B (score 9740) over face A (score 7810)
    even though both have the same area (10000 px²).

    NOTE: This test is RED until _doa_score_for_face is implemented. Current code
    applies a flat 1.2x bonus to ALL faces, giving equal scores — then the first
    candidate wins (face A), not face B.
    """
    detector = EngagementDetector(config, event_bus)

    # Set DOA angle to 0 (front/center of camera field of view)
    event_bus.emit(EventType.DOA_UPDATE, {"angle": 0})
    assert detector._last_doa_angle == 0

    landmarks = np.random.rand(468, 3).astype(np.float32)

    # Face A: left of frame — farther from DOA=0
    # center_x = 50 + 120/2 = 110 → angle ≈ -19.7° from camera center
    face_a = FaceTrack(
        track_id=10,
        bbox=(50, 100, 120, 120),
        landmarks=landmarks,
        head_yaw=0.0,
        head_pitch=0.0,
        last_seen=0.0,
        confidence=0.9,
        is_target=False,
        face_area=10000,
    )

    # Face B: near center frame — close to DOA=0
    # center_x = 300 + 90/2 = 345 → angle ≈ 2.34° from camera center
    face_b = FaceTrack(
        track_id=20,
        bbox=(300, 100, 90, 90),
        landmarks=landmarks,
        head_yaw=0.0,
        head_pitch=0.0,
        last_seen=0.0,
        confidence=0.9,
        is_target=False,
        face_area=10000,
    )

    gaze_a = GazeResult(
        track_id=10,
        yaw_deg=5.0,
        pitch_deg=3.0,
        is_looking_at_robot=True,
        timestamp=0.0,
    )
    gaze_b = GazeResult(
        track_id=20,
        yaw_deg=5.0,
        pitch_deg=3.0,
        is_looking_at_robot=True,
        timestamp=0.0,
    )

    result = detector._select_primary_user(
        [face_a, face_b],
        {10: gaze_a, 20: gaze_b},
    )

    assert result is not None, "Expected a primary user to be selected"
    assert result.track_id == 20, (
        f"Expected face B (track_id=20, near DOA=0, score≈9740) to win over "
        f"face A (track_id=10, off-angle, score≈7810), "
        f"but got track_id={result.track_id}. "
        f"Per-face DOA scoring must prefer angular proximity (equal area tiebreaker)."
    )
