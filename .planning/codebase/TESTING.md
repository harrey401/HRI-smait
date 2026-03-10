# Testing Patterns

**Analysis Date:** 2026-03-09

## Test Framework

**Runner:**
- pytest >= 7.0.0 (dev dependency in `pyproject.toml`)
- pytest-asyncio >= 0.21.0 (for async test support)
- Config in `pyproject.toml` under `[tool.pytest.ini_options]`

**Configuration:**
```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Run Commands:**
```bash
pytest                    # Run all tests (no tests/ directory exists yet)
pytest -v                 # Verbose output
pytest --cov=smait        # Coverage (requires pytest-cov, not in deps)
```

## Current Test State: CRITICAL GAP

**There are NO automated tests in this project.** The `testpaths = ["tests"]` directory does not exist. No `conftest.py`, no `test_*.py` files, no `tests/` directory anywhere in the project source.

**What exists instead:**
1. `run_tests.py` (root) - A manual integration test harness that runs the full HRI system with metrics collection. Not a pytest test suite; requires live hardware (microphone, camera, GPU models, Jackie robot connection).
2. `isaac_sim_tests/` - A domain randomization test framework for Isaac Sim. Requires NVIDIA Isaac Sim runtime. Uses mock classes when Isaac Sim is unavailable.
3. `smait/utils/test_harness.py` - Referenced in `run_tests.py` but file does not exist (import would fail).

## Test Infrastructure That Needs to Be Created

### Directory Structure (recommended)

```
tests/
    conftest.py              # Shared fixtures
    unit/
        test_config.py       # Config loading, merging, env vars
        test_events.py       # EventBus pub/sub, sync/async handlers
        test_protocol.py     # BinaryFrame, MessageSchema
        test_metrics.py      # MetricsTracker timing, counters
        test_eou_detector.py # EOU heuristic, prediction logic
        test_transcriber.py  # Hallucination filter, confidence filter
        test_engagement.py   # State machine transitions
        test_session.py      # Session lifecycle state machine
        test_audio_buffer.py # RawAudioBuffer ring buffer
        test_video_pipeline.py # Frame buffering, JPEG decode
        test_data_logger.py  # Session logging, HRI checklist scoring
    integration/
        test_audio_pipeline.py   # VAD + segment production
        test_dialogue.py         # LLM fallback chain (mocked APIs)
        test_connection.py       # WebSocket server (mocked client)
        test_hri_system.py       # Full system wiring
    e2e/
        test_session_flow.py     # End-to-end session lifecycle
```

### Fixture Recommendations

**Config fixture (for `conftest.py`):**
```python
import pytest
from smait.core.config import Config, reset_config

@pytest.fixture
def config():
    """Provide a fresh default Config for each test."""
    reset_config()
    return Config()

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset config singleton between tests."""
    reset_config()
    yield
    reset_config()
```

**EventBus fixture:**
```python
@pytest.fixture
def event_bus():
    """Provide a fresh EventBus."""
    from smait.core.events import EventBus
    return EventBus()
```

**Audio data fixtures:**
```python
import numpy as np

@pytest.fixture
def silence_audio():
    """16kHz mono silence, 1 second."""
    return np.zeros(16000, dtype=np.int16)

@pytest.fixture
def speech_audio():
    """16kHz mono white noise simulating speech, 1 second."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(16000) * 3000).astype(np.int16)
```

## Testable Components (no GPU/hardware needed)

These components have logic that can be unit tested without ML models or hardware:

### 1. Config System (`smait/core/config.py`)
- `Config.from_file()` — JSON loading and merging
- `Config.from_env()` — environment variable parsing
- `_merge_dict_into_dataclass()` — recursive dict merging
- `get_config()` / `reset_config()` — singleton behavior

### 2. EventBus (`smait/core/events.py`)
- Subscribe/unsubscribe handlers
- Emit to sync handlers
- Emit to async handlers (needs asyncio event loop)
- Error isolation (handler exception does not crash bus)

### 3. Protocol (`smait/connection/protocol.py`)
- `BinaryFrame.from_bytes()` — frame parsing
- `BinaryFrame.pack()` — frame serialization
- `MessageSchema.state()`, `.transcript()`, `.tts_control()` — JSON message construction
- `MessageSchema.parse_text_message()` — JSON parsing

### 4. MetricsTracker (`smait/utils/metrics.py`)
- Timer start/stop and duration calculation
- Rolling average and P95 computation
- Counter increment
- Summary generation

### 5. EOU Detector (`smait/perception/eou_detector.py`)
- `_heuristic_eou()` — rule-based EOU prediction (no model needed)
- `on_silence()` — silence timeout logic
- `update_transcript()` — state management
- Hard cutoff behavior

### 6. Transcriber Filters (`smait/perception/transcriber.py`)
- `_check_filters()` — hallucination detection, confidence thresholds
- `HALLUCINATION_PHRASES` set membership
- Short utterance rejection logic

### 7. Engagement Detector (`smait/perception/engagement.py`)
- State machine: IDLE -> APPROACHING -> ENGAGED -> LOST
- `_is_walking_past()` — area velocity computation
- `_select_primary_user()` — face scoring
- Gaze duration debouncing

### 8. Session Manager (`smait/session/manager.py`)
- State machine: IDLE -> APPROACHING -> ENGAGED -> CONVERSING -> DISENGAGING -> IDLE
- Timeout logic in `check_timeouts()`
- Face lost grace period
- Farewell detection via `_detect_goodbye()` in dialogue manager

### 9. RawAudioBuffer (`smait/sensors/audio_pipeline.py`)
- Ring buffer write and extract
- Time-to-offset interpolation
- Buffer overrun detection

### 10. VideoPipeline (`smait/sensors/video_pipeline.py`)
- Frame buffering
- `get_frame_at()` — temporal lookup with tolerance
- FPS tracking

### 11. DataLogger (`smait/utils/data_logger.py`)
- Session lifecycle (start/end)
- HRI checklist auto-scoring
- Turn tracking
- JSON serialization

### 12. Face Tracker math (`smait/perception/face_tracker.py`)
- `_compute_iou()` — IOU calculation (static method, pure math)
- `_estimate_head_pose()` — head pose from landmarks (static method)

### 13. Dialogue Manager (`smait/dialogue/manager.py`)
- `_detect_goodbye()` — regex-based farewell detection (static method)
- `_trim_history()` — conversation history management
- `_build_messages()` — message list construction

## Mocking Strategy

**ML Models:**
- All ML components check `self._available` before use and return `None` on unavailability
- For testing, simply never call `init_model()` — the component will use its fallback behavior
- For testing WITH model behavior, mock the model attribute:
  ```python
  asr = ParakeetASR(config)
  asr._model = MagicMock()
  asr._available = True
  asr._model.transcribe.return_value = (["hello world"], [])
  ```

**Network/WebSocket:**
- Mock `websockets.serve()` for ConnectionManager tests
- Mock `requests.post()` for Ollama/OpenAI API calls in DialogueManager
- Use `unittest.mock.AsyncMock` for async method mocking

**EventBus integration:**
- Use real EventBus in integration tests — it is lightweight
- Capture emitted events by subscribing a test handler:
  ```python
  captured = []
  event_bus.subscribe(EventType.SPEECH_SEGMENT, lambda data: captured.append(data))
  ```

**What NOT to mock:**
- Config dataclasses (use real Config with modified fields)
- EventBus (use real instance)
- Dataclass results (construct real instances)
- NumPy arrays (use real arrays with known data)

## Isaac Sim Test Framework

**Location:** `isaac_sim_tests/`

**Purpose:** Domain randomization testing for the ASD (Active Speaker Detection) system. This is a separate test framework from pytest, designed for simulation environments.

**Key files:**
- `isaac_sim_tests/run_test.py` — Main test runner with 5 test phases
- `isaac_sim_tests/domain_randomization/` — Audio2Face client, articulation control, AV delay injection, noise mixing
- `isaac_sim_tests/metrics/collector.py` — Metrics collection

**Test phases:**
1. ASD Parameter Tuning — sweep detection thresholds
2. Audio-Visual Delay Tolerance — inject timing offsets
3. Noise Robustness — inject background noise at various levels
4. Articulation Variation — scale lip movement amplitude
5. Multi-Speaker — simultaneous face/voice scenarios

**Mock classes provided:** `MockIsaacSim`, `MockAudio2Face`, `MockSMAIT` for running without Isaac Sim hardware.

## Coverage

**Requirements:** None enforced. No coverage configuration exists.

**Recommended setup (add to `pyproject.toml`):**
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.coverage.run]
source = ["smait"]
omit = ["smait/demo.py"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

**Recommended dev dependencies (add to `pyproject.toml`):**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

## Priority Test Targets

Components ranked by testability and impact:

| Priority | Component | File | Reason |
|----------|-----------|------|--------|
| P0 | Config system | `smait/core/config.py` | Foundation, pure logic, easy to test |
| P0 | EventBus | `smait/core/events.py` | Central communication, easy to test |
| P0 | Protocol | `smait/connection/protocol.py` | Binary frame parsing, pure logic |
| P0 | MetricsTracker | `smait/utils/metrics.py` | Pure computation, no dependencies |
| P1 | EOU heuristic | `smait/perception/eou_detector.py` | Turn-taking accuracy depends on this |
| P1 | Transcriber filters | `smait/perception/transcriber.py` | Hallucination rejection logic |
| P1 | Engagement state machine | `smait/perception/engagement.py` | Core interaction trigger |
| P1 | Session state machine | `smait/session/manager.py` | Session lifecycle correctness |
| P2 | RawAudioBuffer | `smait/sensors/audio_pipeline.py` | Ring buffer correctness |
| P2 | DataLogger | `smait/utils/data_logger.py` | HRI scoring accuracy |
| P2 | DialogueManager | `smait/dialogue/manager.py` | Goodbye detection, history |
| P3 | FaceTracker IOU | `smait/perception/face_tracker.py` | Math correctness |
| P3 | VideoPipeline | `smait/sensors/video_pipeline.py` | Frame temporal lookup |

---

*Testing analysis: 2026-03-09*
