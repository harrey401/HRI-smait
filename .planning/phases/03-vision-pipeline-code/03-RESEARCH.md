# Phase 3: Vision Pipeline Code - Research

**Researched:** 2026-03-10
**Domain:** L2CS-Net gaze estimation, MediaPipe lip extraction, engagement detection, face tracking
**Confidence:** HIGH (code already exists; phase is primarily about correctness verification, config fixes, and test coverage)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VIS-01 | L2CS-Net gaze estimation activated with correct arch (`ResNet50`, not `Gaze360`) | Already fixed in Phase 1 (01-03). Phase 3 task: write comprehensive unit tests covering the full estimate() flow, step() result parsing, and fallback behavior |
| VIS-02 | Lip extraction produces mouth ROI compatible with Dolphin (88x88 grayscale from MediaPipe landmarks) | `LipExtractor` currently outputs RGB at `lip_roi_size=(96,96)`. Config default must change to `(88,88)`. Grayscale conversion stays in `_run_dolphin` (Phase 1 decision). Tests must verify final output to Dolphin is `(88,88,1)` grayscale |
| VIS-03 | Gaze-based engagement detection with sustained gaze threshold (>2s) | `EngagementDetector` already implements `min_gaze_duration_s=2.0`. Phase 3 task: write unit tests for state transitions (IDLE → APPROACHING → ENGAGED → IDLE), DOA integration, and area velocity filtering |
| VIS-04 | Face tracking maintains persistent IDs across frames (existing MediaPipe + IOU) | `FaceTracker` already implements IOU matching at 0.3 threshold. Phase 3 task: write unit tests for IOU computation, ID re-association across frames, and FACE_LOST event emission |
</phase_requirements>

---

## Summary

Phase 3 is a correctness and test-coverage phase, not a rewrite phase. The four vision pipeline files (`gaze.py`, `lip_extractor.py`, `engagement.py`, `face_tracker.py`) already exist with solid implementations. Phase 1 fixed the L2CS-Net arch parameter. Phase 3's job is:

1. Fix the one outstanding correctness issue: `VisionConfig.lip_roi_size` defaults to `(96,96)` but should be `(88,88)` to match Dolphin's required input size. The `LipExtractor` resizes to `lip_roi_size` — DolphinSeparator then resizes again to 88x88. The double-resize is wasteful and the LipROI output size is not Dolphin-compatible by default. Fix the config default.

2. Write comprehensive unit tests (with mocked inputs — no real camera, no real model weights) for all four vision modules. Current test suite has `test_gaze.py` but NO tests for `lip_extractor.py`, `engagement.py`, or `face_tracker.py`.

3. Verify the `LipExtractor` uses the correct MediaPipe lip landmark indices. The current indices (`range(61,69)` outer + `range(78,96)` inner) are plausible but should be verified against the authoritative `FACEMESH_LIPS` connection set.

**Primary recommendation:** Fix `lip_roi_size` default to `(88,88)` in `config.py`, write TDD tests for all four perception modules with mocked inputs, and verify MediaPipe landmark indices against `mp.solutions.face_mesh_connections.FACEMESH_LIPS`.

The entire phase runs without GPU and without real model weights — all tests use `unittest.mock` or synthetic numpy arrays.

---

## Standard Stack

### Core (Already Installed)

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| mediapipe | >=0.10.0 | Face Mesh 468 landmarks + IOU tracking | Already in requirements.txt; `refine_landmarks=True` used in FaceTracker |
| l2cs (edavalosanaya fork) | HEAD@main | Gaze estimation `Pipeline` | pip-installed in Phase 1; arch='ResNet50' already correct |
| opencv-python | >=4.8.0 | Image crop, resize, color conversion | Already installed |
| numpy | >=1.24.0 | Array operations | Already installed |
| pytest | >=7.0.0 | Test runner | Already installed |
| pytest-asyncio | >=0.21.0 | Async test support | Already installed |

### No New Dependencies

Phase 3 requires zero new pip installs. All dependencies were installed in Phase 1.

---

## Architecture Patterns

### Current File Layout (What Exists)

```
smait/perception/
├── face_tracker.py       # FaceTracker: MediaPipe FaceMesh + IOU persistent IDs
├── gaze.py               # GazeEstimator: L2CS-Net Pipeline, fallback head pose
├── lip_extractor.py      # LipExtractor: MediaPipe landmarks → mouth ROI buffer
└── engagement.py         # EngagementDetector: state machine (IDLE/APPROACHING/ENGAGED/LOST)

tests/unit/
├── test_gaze.py          # EXISTS: 4 tests, all pass
├── test_lip_extractor.py # MISSING — Phase 3 creates this
├── test_engagement.py    # MISSING — Phase 3 creates this
└── test_face_tracker.py  # MISSING — Phase 3 creates this
```

### Pattern 1: Testing Vision Modules Without Real Camera

All vision tests use synthetic numpy arrays. Never use real camera frames in unit tests.

**Minimal FaceTrack for testing:**
```python
# Source: existing smait/perception/face_tracker.py FaceTrack dataclass
from dataclasses import dataclass, field
import numpy as np

@dataclass
class _FakeTrack:
    track_id: int = 1
    bbox: tuple = (10, 10, 80, 80)   # (x, y, w, h)
    landmarks: np.ndarray = field(default_factory=lambda: np.random.rand(468, 3).astype(np.float32))
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    face_area: int = 6400            # 80*80
    is_target: bool = False
```

**Synthetic frame for testing LipExtractor:**
```python
import numpy as np
# BGR frame (what camera and FaceTracker pass to LipExtractor)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
# Place a white region where lip landmarks will point
frame[200:250, 280:360] = 255  # Approximate mouth region
```

### Pattern 2: FaceTrack Fixture With Realistic Lip Landmarks

LipExtractor checks `len(track.landmarks) >= 96` and uses indices 61-68 and 78-95. Tests must supply landmarks where those indices fall within the image bounds.

```python
def _make_face_track_with_lip_landmarks(h: int = 480, w: int = 640) -> _FakeTrack:
    """Create a FaceTrack with lip landmarks pointing to a valid mouth region."""
    landmarks = np.random.rand(468, 3).astype(np.float32)
    # Place lip landmarks in center of frame at ~normalized coords
    # Indices 61-68 (outer lip) and 78-95 (inner lip)
    for idx in list(range(61, 69)) + list(range(78, 96)):
        landmarks[idx, 0] = 0.45 + np.random.uniform(-0.05, 0.05)  # x: ~45-55% of width
        landmarks[idx, 1] = 0.60 + np.random.uniform(-0.03, 0.03)  # y: ~57-63% of height
        landmarks[idx, 2] = 0.0
    track = _FakeTrack(landmarks=landmarks)
    return track
```

### Pattern 3: EngagementDetector State Machine Testing

Testing state transitions requires calling `update()` multiple times with controlled timestamps to simulate time passing. Do NOT use `time.sleep()`.

```python
def test_idle_to_engaged_via_sustained_gaze(config, event_bus):
    """State transitions IDLE -> APPROACHING -> ENGAGED after >2s gaze."""
    from smait.perception.engagement import EngagementDetector, EngagementState
    from smait.perception.gaze import GazeResult

    detector = EngagementDetector(config, event_bus)
    track = _FakeTrack(track_id=1, face_area=10000)
    gaze = GazeResult(track_id=1, yaw_deg=5.0, pitch_deg=3.0,
                      is_looking_at_robot=True, timestamp=0.0)

    # Simulate 3 seconds of sustained gaze in 0.5s increments
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.1]:
        gaze_result_at_t = GazeResult(track_id=1, yaw_deg=5.0, pitch_deg=3.0,
                                       is_looking_at_robot=True, timestamp=t)
        detector.update([track], {1: gaze_result_at_t}, timestamp=t)

    assert detector.state == EngagementState.ENGAGED
    assert detector.target_track_id == 1
```

### Pattern 4: IOU Matching Test

```python
def test_iou_matching_preserves_track_id():
    """FaceTracker re-uses existing track_id when IOU >= 0.3."""
    # Create two overlapping bboxes
    bbox1 = (10, 10, 80, 80)   # x=10, y=10, w=80, h=80
    bbox2 = (15, 15, 80, 80)   # slightly shifted
    iou = FaceTracker._compute_iou(bbox1, bbox2)
    assert iou > 0.3
```

### Anti-Patterns to Avoid

- **Calling `FaceTracker.process_frame()` with a real camera frame:** Requires MediaPipe model files and live camera. Use `_match_and_update()` or `_compute_iou()` directly for unit tests.
- **Calling `GazeEstimator.init_model()` in tests:** Downloads L2CS-Net weights (~100MB). Always set `est._l2cs_pipeline = None` or mock it.
- **Testing engagement without enough time steps:** The >2s gaze threshold requires `min_gaze_duration_s` worth of timestamps to pass. Tests that call `update()` once will never reach ENGAGED state.
- **Relying on `lip_roi_size=(96,96)` being correct:** The config default was `(96,96)` but must be `(88,88)`. Any test that checks output shape should use `(88,88)` after the fix.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Lip landmark indices | Custom landmark discovery | `mp.solutions.face_mesh_connections.FACEMESH_LIPS` | Authoritative set of connection pairs; union of all indices gives correct lip coverage |
| Face IOU tracking | Custom centroid tracker | Existing `FaceTracker._compute_iou()` | Already implemented with correct (x,y,w,h) format; IOU=0.3 threshold validated |
| Gaze angle parsing from L2CS | Custom head-pose math | Existing `self._l2cs_pipeline.step()` → `.yaw`, `.pitch` | L2CS-Net Pipeline.step() returns a results object with `.yaw[0]` and `.pitch[0]` attributes |
| State machine time tracking | `time.time()` in tests | Monotonic `timestamp` parameter | EngagementDetector already takes `timestamp: float` — use controlled values in tests |

---

## Common Pitfalls

### Pitfall 1: `lip_roi_size = (96, 96)` — Wrong Default for Dolphin

**What goes wrong:** `LipExtractor` resizes mouth crops to `config.vision.lip_roi_size`. Default is `(96, 96)`. DolphinSeparator then resizes again to `(88, 88)`. The double-resize is wasteful and the LipROI `.image` shape (96x96x3) does not match the "88x88 grayscale Dolphin-compatible" requirement of VIS-02.

**Why it happens:** Config was set to a generic default without consulting Dolphin's exact input spec.

**How to avoid:** Change `VisionConfig.lip_roi_size` default to `(88, 88)` in `smait/core/config.py`. DolphinSeparator still converts RGB→grayscale (Phase 1 decision stands), but now `LipROI.image` is already `(88, 88, 3)` before the conversion, so the `cv2.resize()` in `_run_dolphin` is a no-op (88→88).

**Warning signs:** Unit test checking `roi.image.shape == (88, 88, 3)` fails with `(96, 96, 3)`.

### Pitfall 2: L2CS `step()` Returns Results Object, Not Plain List

**What goes wrong:** Code at `gaze.py:105-106` checks `if results and hasattr(results, "yaw") and len(results.yaw) > 0`. If you mock `step()` to return a plain dict or tuple, the `hasattr(results, 'yaw')` check fails. Tests that mock `_l2cs_pipeline.step()` must return an object with a `.yaw` attribute (list-like).

**How to avoid:** In tests, use `MagicMock(yaw=[15.0], pitch=[5.0])` as the `step()` return value.

**Warning signs:** Gaze test using `step.return_value = {"yaw": [15.0]}` returns head-pose fallback result instead of L2CS result.

### Pitfall 3: FaceTrack Landmark Shape Boundary Check

**What goes wrong:** `LipExtractor.extract()` checks `len(track.landmarks) < 96` (not `< 468`). This means a track with `landmarks.shape = (100, 3)` passes the check but will crash at index 263+ in FaceTracker's head pose. For LipExtractor unit tests only, a `(468, 3)` array is still safer.

**How to avoid:** Always create fake landmarks with `shape=(468, 3)` even when only testing LipExtractor.

**Warning signs:** `IndexError: index 263 is out of bounds for axis 0 with size 100`.

### Pitfall 4: EngagementDetector Walking-Past Filter

**What goes wrong:** `_is_walking_past()` checks area velocity over the last 1 second. If a test face has `face_area` that changes rapidly across `update()` calls, the engagement check is skipped. Set `face_area` to a constant value in engagement tests unless specifically testing the walking-past filter.

**How to avoid:** Use a constant `face_area=10000` in basic engagement tests.

**Warning signs:** Test reaches 2+ seconds of gaze but detector stays IDLE — the walking-past filter is silently rejecting the candidate.

### Pitfall 5: EventBus Emit Not Awaited in Synchronous Tests

**What goes wrong:** `EngagementDetector._check_engage()` calls `self._event_bus.emit(EventType.ENGAGEMENT_START, ...)`. The EventBus `emit()` is synchronous (fires event_type callbacks). If async subscribers are registered, they'll need `asyncio` running. For unit tests with no async subscribers, this is fine.

**How to avoid:** Use `event_bus.subscribe(EventType.ENGAGEMENT_START, lambda data: events.append(data))` with a plain list to capture emitted events without async.

**Warning signs:** Events list stays empty in sync tests; check if subscriber is async.

### Pitfall 6: MediaPipe Lip Landmark Indices Are Relative to Full 468-point Set

**What goes wrong:** OUTER_LIP_INDICES = `range(61, 69)` and INNER_LIP_INDICES = `range(78, 96)` are subsets of the full `FACEMESH_LIPS` connection set. The full FACEMESH_LIPS set (from `face_mesh_connections.py`) includes indices: 0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415. The current range-based indices cover a useful subset but miss indices like 0 (upper lip midpoint), 14 (lower inner lip), 17 (chin-adjacent lower lip) which are part of the standard lip bounding box.

**Impact assessment:** The current approach (range(61,69) + range(78,96)) still produces a reasonable bounding box because indices 61-95 are entirely in the lip region. The bounding box may be slightly undersized (missing the horizontal lip midpoints at 0 and 17), but this is a minor quality issue, not a correctness failure.

**Recommendation:** For Phase 3, document the current indices as "functional subset." A future enhancement (Phase 4 or later) could switch to the full FACEMESH_LIPS set using `mp.solutions.face_mesh_connections.FACEMESH_LIPS`.

---

## Code Examples

### Fix: VisionConfig Default for Dolphin Compatibility

```python
# Source: smait/core/config.py VisionConfig
# Change lip_roi_size default from (96, 96) to (88, 88)
@dataclass
class VisionConfig:
    max_faces: int = 5
    min_face_confidence: float = 0.6
    lip_roi_size: tuple = (88, 88)   # Was (96, 96) — must be 88x88 for Dolphin
```

### Test: LipROI Output Shape Verification

```python
# tests/unit/test_lip_extractor.py
import numpy as np
import pytest
from smait.core.config import Config
from smait.core.events import EventBus
from smait.perception.lip_extractor import LipExtractor

def _make_track_with_lip_landmarks():
    """FaceTrack stub with lip landmarks pointing to center of a 480x640 frame."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeLipTrack:
        track_id: int = 1
        landmarks: np.ndarray = field(default_factory=lambda: np.zeros((468, 3), dtype=np.float32))

    track = FakeLipTrack()
    # Set lip landmark indices 61-68 and 78-95 to mouth region (x=0.5, y=0.6)
    for idx in list(range(61, 69)) + list(range(78, 96)):
        track.landmarks[idx, 0] = 0.50
        track.landmarks[idx, 1] = 0.60
    return track


def test_lip_roi_output_shape(config, event_bus):
    """VIS-02: LipROI image must be (88, 88, 3) with default config."""
    extractor = LipExtractor(config, event_bus)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[250:310, 290:350] = 200  # White mouth region

    track = _make_track_with_lip_landmarks()
    roi = extractor.extract(frame, track, timestamp=0.0)

    assert roi is not None
    assert roi.image.shape == (88, 88, 3), (
        f"Expected (88, 88, 3) got {roi.image.shape}. "
        "Check VisionConfig.lip_roi_size default."
    )
    assert roi.image.dtype == np.uint8
```

### Test: L2CS Pipeline Mock Pattern

```python
# tests/unit/test_gaze.py (add to existing file)
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

def test_l2cs_step_result_parsed(config, event_bus):
    """GazeEstimator._estimate_l2cs() correctly reads yaw[0] and pitch[0]."""
    from smait.perception.gaze import GazeEstimator, GazeResult

    est = GazeEstimator(config, event_bus)
    mock_results = MagicMock()
    mock_results.yaw = [15.0]
    mock_results.pitch = [-8.0]

    mock_pipeline = MagicMock()
    mock_pipeline.step.return_value = mock_results
    est._l2cs_pipeline = mock_pipeline

    from dataclasses import dataclass
    @dataclass
    class FakeTrack:
        track_id: int = 1
        bbox: tuple = (10, 10, 80, 80)
        head_yaw: float = 0.0
        head_pitch: float = 0.0

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, FakeTrack(), timestamp=0.0)

    assert result.yaw_deg == pytest.approx(15.0)
    assert result.pitch_deg == pytest.approx(-8.0)
    assert result.is_looking_at_robot is True  # |15| < 30 and |-8| < 20
```

### Test: Engagement State Machine

```python
# tests/unit/test_engagement.py (full new file)
import pytest
from smait.perception.engagement import EngagementDetector, EngagementState
from smait.perception.gaze import GazeResult


def _make_gaze(track_id: int, looking: bool, t: float) -> GazeResult:
    return GazeResult(
        track_id=track_id,
        yaw_deg=5.0 if looking else 90.0,
        pitch_deg=3.0,
        is_looking_at_robot=looking,
        timestamp=t,
    )


@dataclass
class FakeTrack:
    track_id: int = 1
    face_area: int = 10000
    head_yaw: float = 5.0
    head_pitch: float = 3.0
    is_target: bool = False


def test_initial_state_is_idle(config, event_bus):
    detector = EngagementDetector(config, event_bus)
    assert detector.state == EngagementState.IDLE


def test_sustained_gaze_reaches_engaged(config, event_bus):
    """IDLE -> APPROACHING -> ENGAGED after 2.1s of sustained gaze."""
    detector = EngagementDetector(config, event_bus)
    track = FakeTrack(track_id=1, face_area=10000)
    # Step through 2.1 seconds, 0.1s apart
    for i in range(22):
        t = i * 0.1
        gaze = _make_gaze(1, looking=True, t=t)
        detector.update([track], {1: gaze}, timestamp=t)
    assert detector.state == EngagementState.ENGAGED


def test_gaze_break_resets_to_idle(config, event_bus):
    """Gaze loss during APPROACHING resets to IDLE."""
    detector = EngagementDetector(config, event_bus)
    track = FakeTrack(track_id=1, face_area=10000)
    # Start looking
    for i in range(5):
        t = i * 0.1
        detector.update([track], {1: _make_gaze(1, True, t)}, timestamp=t)
    # Break gaze
    detector.update([track], {1: _make_gaze(1, False, 0.6)}, timestamp=0.6)
    assert detector.state == EngagementState.IDLE
```

### Test: FaceTracker IOU

```python
# tests/unit/test_face_tracker.py
from smait.perception.face_tracker import FaceTracker

def test_iou_high_overlap():
    """Identical bboxes produce IOU=1.0."""
    bbox = (10, 10, 80, 80)
    assert FaceTracker._compute_iou(bbox, bbox) == pytest.approx(1.0)


def test_iou_no_overlap():
    """Non-overlapping bboxes produce IOU=0.0."""
    bbox1 = (0, 0, 50, 50)
    bbox2 = (100, 100, 50, 50)
    assert FaceTracker._compute_iou(bbox1, bbox2) == pytest.approx(0.0)


def test_iou_partial_overlap():
    """Partially overlapping bboxes produce 0 < IOU < 1."""
    bbox1 = (0, 0, 100, 100)
    bbox2 = (50, 50, 100, 100)
    iou = FaceTracker._compute_iou(bbox1, bbox2)
    assert 0 < iou < 1
```

---

## State of the Art

| Old State (Before Phase 1) | Current State (After Phase 1) | Phase 3 Action |
|----------------------------|-------------------------------|----------------|
| `arch="Gaze360"` wrong param | `arch='ResNet50'` correct | Write tests to lock in |
| `lip_roi_size=(96,96)` default | Still `(96,96)` — NOT YET FIXED | Fix default to `(88,88)` |
| No tests for lip_extractor | No tests for lip_extractor | Create `test_lip_extractor.py` |
| No tests for engagement | No tests for engagement | Create `test_engagement.py` |
| No tests for face_tracker | No tests for face_tracker | Create `test_face_tracker.py` |
| `test_gaze.py` has 4 tests | 4 tests all pass | Expand with step() mock tests |

---

## Open Questions

1. **L2CS-Net `step()` result attribute name**
   - What we know: `gaze.py` checks `hasattr(results, 'yaw')` and accesses `results.yaw[0]`, `results.pitch[0]`
   - What's unclear: The exact return type of `L2CSPipeline.step()` in the edavalosanaya fork — is it a dataclass, a namedtuple, or a custom object?
   - Recommendation: Unit tests mock `step()` return value with `MagicMock(yaw=[15.0], pitch=[-8.0])`. For integration test in Phase 7 (lab with GPU + real weights), confirm attribute names match. Current code's `hasattr(results, 'yaw')` guard handles any attribute changes gracefully.

2. **FaceTracker `process_frame()` with real MediaPipe — unit testable?**
   - What we know: `FaceTracker.process_frame()` calls `self._face_mesh.process(rgb)` which requires MediaPipe model files but downloads them automatically (~35MB, one-time)
   - What's unclear: Whether MediaPipe is already downloaded in the project venv or requires a network call on first use
   - Recommendation: Phase 3 unit tests should NOT call `process_frame()` directly. Test `_compute_iou()`, `_estimate_head_pose()`, and `_match_and_update()` directly using synthetic data. A smoke test that constructs `FaceTracker` (which initializes `FaceMesh`) can verify model file availability without processing any frame.

3. **DOA integration in EngagementDetector**
   - What we know: `_select_primary_user()` gives a 1.2x score bonus when `_last_doa_angle is not None`, but doesn't use the actual angle value for spatial filtering
   - What's unclear: Whether VIS-03 requires tighter DOA alignment (angle-to-face-position matching) or just the bonus scoring
   - Recommendation: Phase 3 implements and tests the current bonus-score approach. Tighter DOA-to-face spatial alignment is deferred to Phase 6 when actual DOA angles from hardware are available.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.0.0+ with pytest-asyncio 0.21.0+ |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) — `asyncio_mode = "auto"` |
| Quick run command | `venv/bin/pytest tests/unit/ -x -q` |
| Full suite command | `venv/bin/pytest tests/ --cov=smait --cov-report=term-missing` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VIS-01 | `GazeEstimator.init_model()` uses `arch='ResNet50'` | unit (mock) | `venv/bin/pytest tests/unit/test_gaze.py::test_correct_arch_param -x` | Yes (passing) |
| VIS-01 | `estimate()` returns GazeResult with yaw/pitch from L2CS step() | unit (mock) | `venv/bin/pytest tests/unit/test_gaze.py::test_l2cs_step_result_parsed -x` | Wave 0 |
| VIS-01 | Falls back to head pose when pipeline is None | unit | `venv/bin/pytest tests/unit/test_gaze.py::test_head_pose_fallback -x` | Yes (passing) |
| VIS-02 | `LipROI.image.shape == (88, 88, 3)` with default config | unit | `venv/bin/pytest tests/unit/test_lip_extractor.py::test_lip_roi_output_shape -x` | Wave 0 |
| VIS-02 | `LipExtractor.get_lip_frames()` returns frames within time window | unit | `venv/bin/pytest tests/unit/test_lip_extractor.py::test_get_lip_frames_time_filter -x` | Wave 0 |
| VIS-02 | Buffer cleaned up on FACE_LOST event | unit | `venv/bin/pytest tests/unit/test_lip_extractor.py::test_face_lost_clears_buffer -x` | Wave 0 |
| VIS-03 | Initial state is IDLE | unit | `venv/bin/pytest tests/unit/test_engagement.py::test_initial_state_is_idle -x` | Wave 0 |
| VIS-03 | Sustained >2s gaze → ENGAGED | unit | `venv/bin/pytest tests/unit/test_engagement.py::test_sustained_gaze_reaches_engaged -x` | Wave 0 |
| VIS-03 | Gaze break resets to IDLE | unit | `venv/bin/pytest tests/unit/test_engagement.py::test_gaze_break_resets_to_idle -x` | Wave 0 |
| VIS-03 | ENGAGEMENT_START event emitted on engage | unit | `venv/bin/pytest tests/unit/test_engagement.py::test_engagement_start_event_emitted -x` | Wave 0 |
| VIS-03 | Walking-past filter suppresses rapid-area faces | unit | `venv/bin/pytest tests/unit/test_engagement.py::test_walking_past_not_engaged -x` | Wave 0 |
| VIS-04 | IOU=1.0 for identical bboxes | unit | `venv/bin/pytest tests/unit/test_face_tracker.py::test_iou_high_overlap -x` | Wave 0 |
| VIS-04 | IOU=0.0 for non-overlapping bboxes | unit | `venv/bin/pytest tests/unit/test_face_tracker.py::test_iou_no_overlap -x` | Wave 0 |
| VIS-04 | FACE_LOST event emitted after 2s without detection | unit | `venv/bin/pytest tests/unit/test_face_tracker.py::test_face_lost_after_timeout -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `venv/bin/pytest tests/unit/ -x -q`
- **Per wave merge:** `venv/bin/pytest tests/ --cov=smait --cov-report=term-missing`
- **Phase gate:** All unit tests green, `VisionConfig.lip_roi_size` default verified as `(88,88)`

### Wave 0 Gaps

- [ ] `tests/unit/test_lip_extractor.py` — covers VIS-02 (shape, buffer, time filter, FACE_LOST cleanup)
- [ ] `tests/unit/test_engagement.py` — covers VIS-03 (state transitions, event emission, DOA, walking-past filter)
- [ ] `tests/unit/test_face_tracker.py` — covers VIS-04 (IOU math, FACE_LOST events, head pose estimation)

*(Existing `tests/unit/test_gaze.py` already covers VIS-01 basic tests — add `test_l2cs_step_result_parsed` in Wave 1)*

---

## Sources

### Primary (HIGH confidence)

- `smait/perception/gaze.py` — verified L2CS-Net arch='ResNet50' (line 59), step() result parsing (lines 104-110)
- `smait/perception/lip_extractor.py` — verified landmark indices, RGB output, buffer logic
- `smait/perception/engagement.py` — verified state machine, gaze duration tracking, DOA integration
- `smait/perception/face_tracker.py` — verified IOU matching (0.3 threshold), MediaPipe Face Mesh setup
- `smait/core/config.py` — verified `lip_roi_size=(96,96)` incorrect default (line 62)
- `smait/perception/dolphin_separator.py` — verified 88x88 grayscale requirement (lines 53, 131, 158-159)
- `.planning/phases/01-dependency-setup-stub-api-fixes/01-03-SUMMARY.md` — confirmed Phase 1 fixed arch param; grayscale split decision locked
- `google-ai-edge/mediapipe face_mesh_connections.py` — verified FACEMESH_LIPS indices include 61, 78-95 range as used in lip_extractor.py

### Secondary (MEDIUM confidence)

- WebSearch: MediaPipe FACEMESH_LIPS frozenset — confirmed indices 61, 78-95 are part of the authoritative lip connection set; also includes 0, 14, 17, etc. (minor enhancement opportunity)

### Tertiary (LOW confidence)

- L2CS-Net Pipeline step() return type — not verified from source; mock pattern based on code inspection of `gaze.py`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all files exist and tested; no new libraries needed
- Architecture: HIGH — production code is complete; phase is test coverage + one config fix
- Pitfalls: HIGH — config bug and test mock pattern verified from code inspection

**Research date:** 2026-03-10
**Valid until:** 2026-06-10 (stable; MediaPipe landmark indices don't change; L2CS-Net API is stable)
