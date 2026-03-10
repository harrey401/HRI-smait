"""Test scaffold for GazeEstimator — RED phase (Plan 01).

Head pose fallback test PASSES now.
Import-correctness test is marked xfail until Plan 03 fixes the arch param.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from smait.core.config import Config
from smait.core.events import EventBus
from smait.perception.gaze import GazeEstimator, GazeResult


@dataclass
class _FakeTrack:
    """Minimal FaceTrack stub for testing without real detection."""
    track_id: int = 1
    bbox: tuple = (10, 10, 80, 80)
    head_yaw: float = 0.0
    head_pitch: float = 0.0


@pytest.fixture
def gaze_estimator(config, event_bus):
    """GazeEstimator without init_model() — _l2cs_pipeline not set."""
    est = GazeEstimator(config, event_bus)
    # Ensure l2cs_pipeline is None (no model loaded)
    if not hasattr(est, "_l2cs_pipeline"):
        est._l2cs_pipeline = None
    return est


def test_head_pose_fallback(gaze_estimator):
    """GazeEstimator falls back to head pose when L2CS-Net is not available."""
    import numpy as np

    track = _FakeTrack(track_id=1, head_yaw=5.0, head_pitch=3.0)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = gaze_estimator.estimate(image, track, timestamp=1.0)

    assert isinstance(result, GazeResult)
    assert result.track_id == 1
    assert result.yaw_deg == 5.0
    assert result.pitch_deg == 3.0


def test_head_pose_looking_at_robot(gaze_estimator):
    """is_looking_at_robot is True when yaw/pitch are within thresholds."""
    import numpy as np

    track = _FakeTrack(track_id=2, head_yaw=10.0, head_pitch=5.0)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = gaze_estimator.estimate(image, track, timestamp=0.0)

    assert result.is_looking_at_robot is True


def test_head_pose_not_looking_at_robot(gaze_estimator):
    """is_looking_at_robot is False when yaw exceeds threshold."""
    import numpy as np

    track = _FakeTrack(track_id=3, head_yaw=45.0, head_pitch=5.0)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = gaze_estimator.estimate(image, track, timestamp=0.0)

    assert result.is_looking_at_robot is False


@pytest.mark.xfail(reason="Stub not yet fixed - Plan 03: gaze.py still uses arch='Gaze360'")
def test_correct_arch_param(config, event_bus):
    """QUAL-01: init_model() must pass arch='ResNet50', not arch='Gaze360'.

    This test is RED — it will fail until Plan 03 fixes gaze.py to use
    'arch=\\'ResNet50\\'' in the L2CSPipeline constructor call.
    """
    mock_pipeline_cls = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    with patch("l2cs.Pipeline", mock_pipeline_cls):
        est = GazeEstimator(config, event_bus)
        asyncio.run(est.init_model())

    call_kwargs = mock_pipeline_cls.call_args
    arch_value = call_kwargs.kwargs.get("arch") or call_kwargs.args[1] if call_kwargs.args else None
    # Check keyword arg
    assert call_kwargs.kwargs.get("arch") == "ResNet50", (
        f"Expected arch='ResNet50', got arch='{call_kwargs.kwargs.get('arch')}'"
    )
