# Coding Conventions

**Analysis Date:** 2026-03-09

## Naming Patterns

**Files:**
- Use `snake_case.py` for all modules: `audio_pipeline.py`, `face_tracker.py`, `data_logger.py`
- One primary class per module, named after the file: `audio_pipeline.py` contains `AudioPipeline`
- Supporting dataclasses co-located in the same file as the class that produces them

**Functions:**
- Use `snake_case` for all functions and methods
- Private methods prefixed with underscore: `_emit_segment()`, `_check_approach()`
- Event handlers prefixed with `_on_`: `_on_tts_start()`, `_on_face_lost()`, `_on_engagement_start()`
- Async methods use `async def` with `await` — no callback-based async patterns
- Properties via `@property` decorator for simple getters: `available`, `state`, `connected`, `tracks`

**Variables:**
- Private instance attributes prefixed with underscore: `self._config`, `self._model`, `self._available`
- Public attributes without underscore only for composed sub-systems on `HRISystem`: `self.event_bus`, `self.metrics`, `self.dialogue`
- Module-level constants in `UPPER_SNAKE_CASE`: `MAX_LOOP_RETRIES`, `FACE_LOST_TIMEOUT_S`, `IOU_THRESHOLD`
- Module-level constants defined at top of file, after imports, before classes

**Types:**
- Classes use `PascalCase`: `AudioPipeline`, `SpeechSegment`, `ConnectionManager`
- Enums use `PascalCase` with `UPPER_SNAKE_CASE` members: `EventType.SPEECH_DETECTED`, `SessionState.IDLE`, `FrameType.AUDIO_CAE`
- Dataclasses for all data transfer objects: `SpeechSegment`, `TranscriptResult`, `GazeResult`, `SeparationResult`, `TurnLog`, `SessionLog`

**Packages:**
- Use `snake_case` for package directories: `smait/perception/`, `smait/sensors/`, `smait/connection/`

## Code Style

**Formatting:**
- Black formatter, line length 100
- Configured in `pyproject.toml` under `[tool.black]`
- Target versions: py310, py311, py312

**Linting:**
- Ruff linter, line length 100
- Rules: E (pycodestyle errors), F (pyflakes), W (pycodestyle warnings), I (isort), N (pep8-naming), UP (pyupgrade)
- E501 ignored (line length handled by Black)
- Configured in `pyproject.toml` under `[tool.ruff]`

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first, in every module)
2. Standard library: `asyncio`, `logging`, `time`, `json`, `os`, `re`, `uuid`
3. Third-party: `numpy`, `torch`, `cv2`, `mediapipe`, `websockets`
4. Local project: `from smait.core.config import Config`

**Patterns:**
- Prefer `from X import Y` over `import X` for project modules
- Use explicit imports — never `from smait.perception import *`
- Lazy imports for heavy ML libraries inside methods to avoid import-time GPU loading:
  ```python
  # In smait/perception/asr.py
  import nemo.collections.asr as nemo_asr  # inside init_model(), not at top level
  ```
- Type-ignore comments for optional ML packages: `# type: ignore[import-not-found]`

**Path Aliases:**
- None used. All imports are relative to project root via `smait.*` namespace

## `__init__.py` Patterns

- Package `__init__.py` files re-export public API classes with explicit `__all__` lists
- Example from `smait/perception/__init__.py`:
  ```python
  from .face_tracker import FaceTracker, FaceTrack
  from .gaze import GazeEstimator, GazeResult
  __all__ = ["FaceTracker", "FaceTrack", "GazeEstimator", "GazeResult", ...]
  ```
- Root `smait/__init__.py` contains only version and docstring:
  ```python
  __version__ = "3.0.0"
  ```

## Type Annotations

**Patterns:**
- Use `from __future__ import annotations` in every file for PEP 604 union syntax
- Return type annotations on all public methods: `async def start(self) -> None:`
- Parameter type annotations on all public methods: `def transcribe(self, audio: np.ndarray, ...) -> Optional[TranscriptResult]:`
- Use `Optional[X]` (from `typing`) for nullable types, not `X | None` — despite future annotations
- Use `list[X]`, `dict[K, V]`, `tuple[X, Y]` (lowercase) for generic types
- Use `Any` from typing when event data is polymorphic (event handler signatures)

## Dataclass Patterns

**When to use:**
- All data transfer objects / result types: `TranscriptResult`, `SpeechSegment`, `GazeResult`, `SeparationResult`, `DialogueResponse`
- Configuration groups: `AudioConfig`, `VisionConfig`, `ConnectionConfig`
- Logged records: `TurnLog`, `SessionLog`, `HRIChecklist`

**Pattern:**
```python
from dataclasses import dataclass, field

@dataclass
class SpeechSegment:
    """A complete speech segment with aligned audio from both streams."""
    cae_audio: np.ndarray
    raw_audio: Optional[np.ndarray]
    start_time: float
    end_time: float
    duration: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration = self.end_time - self.start_time
```

- Computed fields use `field(init=False)` with `__post_init__`
- Computed properties use `@property` for derived values: `HRIChecklist.score`, `FaceTrack.center`
- Nested dataclass composition uses `field(default_factory=...)`: `Config.connection: ConnectionConfig = field(default_factory=ConnectionConfig)`

## Error Handling

**Strategy: Graceful Degradation**
- Every ML model component has an `_available` boolean flag checked before use
- If a model fails to load (ImportError or runtime error), the system continues with reduced functionality
- Pattern used in: `smait/perception/asr.py`, `smait/perception/dolphin_separator.py`, `smait/perception/gaze.py`, `smait/perception/eou_detector.py`, `smait/output/tts.py`

**Model Loading Pattern:**
```python
async def init_model(self) -> None:
    try:
        from some_ml_lib import Model  # type: ignore[import-not-found]
        self._model = Model.from_pretrained(...)
        self._available = True
        logger.info("Model loaded")
    except ImportError:
        logger.warning("Package not installed. Feature unavailable.")
    except Exception:
        logger.exception("Failed to load model")
```

**Runtime Error Pattern:**
```python
try:
    result = self._model.transcribe(...)
except Exception:
    logger.exception("ASR transcription failed")
    return None
```

**Resilient Loops:**
- `HRISystem._resilient_loop()` in `smait/main.py` wraps async loops with retry logic
- Max 10 retries with 1-second cooldown between crashes
- `asyncio.CancelledError` breaks the loop (graceful shutdown), all other exceptions retry

**Fallback Chains:**
- Dialogue: Ollama local -> OpenAI API -> static fallback message (`smait/dialogue/manager.py`)
- TTS: Kokoro GPU -> Android TTS text fallback (`smait/output/tts.py`)
- Gaze: L2CS-Net -> head pose estimation from landmarks (`smait/perception/gaze.py`)
- Separation: Dolphin AV-TSE -> CAE audio passthrough (`smait/perception/dolphin_separator.py`)
- EOU: LiveKit model -> transformers model -> heuristic rules (`smait/perception/eou_detector.py`)

## Logging

**Framework:** Python `logging` module

**Patterns:**
- Module-level logger: `logger = logging.getLogger(__name__)` at top of every file
- Log levels used consistently:
  - `logger.info()` for state transitions, model loading, key pipeline events
  - `logger.debug()` for per-frame / per-chunk data
  - `logger.warning()` for degraded states (model not loaded, feature disabled)
  - `logger.exception()` for caught exceptions (auto-includes traceback)
- Format configured in `run_jackie.py`: `"%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"`
- Dual output: stdout + `smait.log` file
- Noisy third-party loggers suppressed: `websockets`, `mediapipe`, `nemo`, `torch`, `urllib3`

## Docstrings

**Module-level:** Every `.py` file starts with a one-line docstring describing purpose:
```python
"""Silero VAD + ring buffer -> SpeechSegment production."""
```

**Class-level:** Multi-line docstrings with architecture context, algorithms used, and usage notes:
```python
class DolphinSeparator:
    """Dolphin Audio-Visual Target Speaker Extraction.

    This is the core innovation -- it simultaneously:
    1. Identifies who is speaking (via lip-audio correlation)
    2. Separates their voice from the mix (source separation)
    ...
    """
```

**Method-level:** Google-style docstrings with Args/Returns for public methods:
```python
def transcribe(self, audio: np.ndarray, ...) -> Optional[TranscriptResult]:
    """Transcribe a clean audio segment.

    Args:
        audio: float32 mono audio at 16kHz
        ...

    Returns:
        TranscriptResult or None if ASR is unavailable.
    """
```

## Module Design

**Exports:**
- Each package's `__init__.py` re-exports its public classes
- `__all__` always defined in `__init__.py` files

**Constructor Pattern:**
- All major classes accept `(config: Config, event_bus: EventBus)` as first two args
- Exception: `ParakeetASR` only takes `Config` (no event bus, wrapped by `Transcriber`)
- Components store only their relevant config section: `self._config = config.audio`
- Event subscriptions registered in `__init__` or dedicated `_wire_events()` method

**Initialization Pattern (two-phase):**
1. `__init__()` — lightweight, stores config, creates empty state
2. `async init_model()` — heavy GPU loading, network connections, model downloads
- This separation allows the system to construct all components synchronously, then load models asynchronously with progress logging

## Async Patterns

**Event-driven architecture:**
- `EventBus` in `smait/core/events.py` is the central pub/sub mechanism
- All inter-module communication via events, never direct method calls between peer modules
- Event handlers can be sync or async — `EventBus.emit()` detects and schedules appropriately
- `EventBus.emit()` is fire-and-forget; `EventBus.emit_async()` awaits all handlers

**Async loop pattern:**
```python
async def _audio_loop(self) -> None:
    while self._running:
        # do work
        await asyncio.sleep(0.05)  # 50ms check interval
```

**Thread-to-async bridge:**
- Blocking operations wrapped in `run_in_executor()`:
  ```python
  resp = await asyncio.get_event_loop().run_in_executor(
      None,
      lambda: requests.post(...),
  )
  ```

## Configuration Pattern

**Hierarchical dataclass config** in `smait/core/config.py`:
- Top-level `Config` contains nested config dataclasses for each subsystem
- Singleton via `get_config()` / `reset_config()` functions
- Three loading modes: defaults (dataclass defaults), JSON file (`Config.from_file()`), env vars (`Config.from_env()`)
- Env var pattern: `SMAIT_<SECTION>_<FIELD>` e.g., `SMAIT_CONNECTION_PORT=9000`
- CLI overrides applied after config loading in `run_jackie.py`

## Constants and Magic Numbers

- Always extract to named constants at module level
- Issue references in comments: `# Issue #6 mitigation`, `# Issue #1`
- Threshold constants co-located with the class that uses them:
  - `IOU_THRESHOLD = 0.3` in `smait/perception/face_tracker.py`
  - `AREA_VELOCITY_THRESHOLD = 5000` in `smait/perception/engagement.py`
  - `MIN_SEGMENT_DURATION_S = 0.5` in `smait/sensors/audio_pipeline.py`

---

*Convention analysis: 2026-03-09*
