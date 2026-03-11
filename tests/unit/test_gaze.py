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

try:
    import l2cs  # noqa: F401
    _has_l2cs = True
except ImportError:
    _has_l2cs = False


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


@pytest.mark.skipif(not _has_l2cs, reason="l2cs not installed")
def test_correct_arch_param(config, event_bus):
    """QUAL-01: init_model() must pass arch='ResNet50', not arch='Gaze360'."""
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


def test_l2cs_step_result_parsed(config, event_bus):
    """estimate() parses L2CS step() results: yaw=[15.0], pitch=[-8.0] -> GazeResult."""
    import numpy as np

    est = GazeEstimator(config, event_bus)
    est._l2cs_pipeline = MagicMock()
    est._l2cs_pipeline.step.return_value = MagicMock(yaw=[15.0], pitch=[-8.0])

    track = _FakeTrack(track_id=10, head_yaw=0.0, head_pitch=0.0)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = est.estimate(image, track, timestamp=0.0)

    assert isinstance(result, GazeResult)
    assert result.yaw_deg == 15.0, f"Expected yaw_deg=15.0, got {result.yaw_deg}"
    assert result.pitch_deg == -8.0, f"Expected pitch_deg=-8.0, got {result.pitch_deg}"
    # |15| < 30 and |-8| < 20 => is_looking_at_robot must be True
    assert result.is_looking_at_robot is True, (
        f"Expected is_looking_at_robot=True for yaw=15, pitch=-8"
    )


def test_l2cs_step_empty_result_falls_back(config, event_bus):
    """estimate() falls back to head pose when step() returns yaw=[], pitch=[]."""
    import numpy as np

    est = GazeEstimator(config, event_bus)
    est._l2cs_pipeline = MagicMock()
    est._l2cs_pipeline.step.return_value = MagicMock(yaw=[], pitch=[])

    track = _FakeTrack(track_id=11, head_yaw=20.0, head_pitch=5.0)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = est.estimate(image, track, timestamp=0.0)

    assert result.yaw_deg == track.head_yaw, (
        f"Expected yaw_deg={track.head_yaw} (head pose fallback), got {result.yaw_deg}"
    )
    assert result.pitch_deg == track.head_pitch


def test_l2cs_step_exception_falls_back(config, event_bus):
    """estimate() falls back to head pose when step() raises RuntimeError."""
    import numpy as np

    est = GazeEstimator(config, event_bus)
    est._l2cs_pipeline = MagicMock()
    est._l2cs_pipeline.step.side_effect = RuntimeError("GPU OOM")

    track = _FakeTrack(track_id=12, head_yaw=-10.0, head_pitch=2.0)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = est.estimate(image, track, timestamp=0.0)

    assert result.yaw_deg == track.head_yaw, (
        f"Expected yaw_deg={track.head_yaw} (head pose fallback), got {result.yaw_deg}"
    )
    assert result.pitch_deg == track.head_pitch


def test_install_instruction_updated():
    """QUAL-01: ImportError warning must reference edavalosanaya fork, not Ahmednull."""
    import pathlib
    gaze_src = pathlib.Path(__file__).parent.parent.parent / "smait" / "perception" / "gaze.py"
    source = gaze_src.read_text(encoding="utf-8")

    assert "edavalosanaya" in source, (
        "gaze.py install warning must reference edavalosanaya fork, not Ahmednull"
    )
    assert "Ahmednull" not in source, (
        "gaze.py must not reference Ahmednull fork"
    )
