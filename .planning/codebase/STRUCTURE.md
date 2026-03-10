# Codebase Structure

**Analysis Date:** 2026-03-09

## Directory Layout

```
SMAIT-v3/
├── smait/                      # Main Python package (HRI system)
│   ├── __init__.py             # Package version (__version__ = "3.0.0")
│   ├── main.py                 # HRISystem orchestrator class
│   ├── demo.py                 # Individual component demo scripts
│   ├── core/                   # Configuration and event system
│   │   ├── __init__.py
│   │   ├── config.py           # Nested dataclass config + singleton
│   │   └── events.py           # EventBus pub/sub + EventType enum
│   ├── connection/             # WebSocket server + wire protocol
│   │   ├── __init__.py         # Exports ConnectionManager, FrameType, MessageSchema
│   │   ├── manager.py          # WebSocket server, frame demuxing, send methods
│   │   └── protocol.py         # Binary frame types, JSON message schemas
│   ├── sensors/                # Raw data ingestion pipelines
│   │   ├── __init__.py
│   │   ├── audio_pipeline.py   # Silero VAD + ring buffer + mic gating
│   │   └── video_pipeline.py   # JPEG decode + frame buffer
│   ├── perception/             # ML-based understanding
│   │   ├── __init__.py
│   │   ├── face_tracker.py     # MediaPipe Face Mesh + IOU tracking
│   │   ├── lip_extractor.py    # Mouth ROI cropping for Dolphin
│   │   ├── gaze.py             # L2CS-Net gaze estimation
│   │   ├── engagement.py       # Multi-signal engagement detection
│   │   ├── dolphin_separator.py# Dolphin AV-TSE speaker separation
│   │   ├── asr.py              # NVIDIA Parakeet TDT ASR
│   │   ├── transcriber.py      # ASR orchestration + hallucination filter
│   │   └── eou_detector.py     # LiveKit End-of-Utterance model
│   ├── dialogue/               # LLM response generation
│   │   ├── __init__.py
│   │   └── manager.py          # Hybrid Ollama + OpenAI dialogue
│   ├── output/                 # Speech synthesis
│   │   ├── __init__.py
│   │   └── tts.py              # Kokoro-82M TTS with streaming
│   ├── session/                # Interaction lifecycle
│   │   ├── __init__.py
│   │   └── manager.py          # 5-state session FSM
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── data_logger.py      # Structured JSON session logs + WAV saving
│       └── metrics.py          # Latency/performance metrics tracker
├── isaac_sim_tests/            # NVIDIA Isaac Sim test framework
│   ├── run_test.py             # Test runner (5 phases)
│   ├── setup.py                # Isaac Sim test setup
│   ├── config.yaml             # Test configuration
│   ├── README.md               # Isaac Sim test docs
│   ├── domain_randomization/   # Test parameter variation
│   │   ├── __init__.py
│   │   ├── articulation.py     # Facial articulation scaling
│   │   ├── audio2face_client.py# NVIDIA Audio2Face client
│   │   ├── av_delay.py         # Audio-visual delay injection
│   │   └── noise_mixer.py      # Noise injection for robustness testing
│   └── metrics/                # Test metrics collection
│       ├── __init__.py
│       └── collector.py        # Test results aggregation
├── scripts/                    # Standalone utility scripts
│   └── track_metrics.py        # SQLite-based metrics tracking CLI
├── docs/                       # Documentation
│   ├── CAE_INTEGRATION_GUIDE.md# iFLYTEK CAE SDK integration guide
│   ├── isaac_sim_setup.md      # Isaac Sim environment setup
│   ├── spec-page-*.png         # Hardware specification images
│   └── hardware-sdk/           # Reference SDK code (Android/CAE)
├── run_jackie.py               # Primary entry point
├── run_tests.py                # Test harness runner (v2 references)
├── pyproject.toml              # Project metadata, dependencies, tool config
├── requirements.txt            # Pip dependencies with install notes
├── .env.example                # Environment variable template
├── .env                        # Environment variables (not committed)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview and setup guide
├── DEMO_FLOW.md                # Live demo playbook
├── INTERVIEW_GUIDE.md          # Interview preparation
├── REBUILD_PROMPT.md           # System rebuild instructions
├── REBUILD_PROMPT_FINAL.md     # Final rebuild prompt
└── .planning/                  # Planning documents
    └── codebase/               # Codebase analysis (this directory)
```

## Directory Purposes

**`smait/core/`:**
- Purpose: Foundation layer shared by all modules
- Contains: Configuration dataclass hierarchy (`Config` with 12 nested sections), `EventBus` pub/sub system, `EventType` enum
- Key files: `config.py` (singleton config, 220 lines), `events.py` (EventBus + 29 event types, 147 lines)

**`smait/connection/`:**
- Purpose: WebSocket server handling Jackie robot communication
- Contains: Server lifecycle, binary frame demuxing, outbound send helpers, wire protocol definitions
- Key files: `manager.py` (226 lines), `protocol.py` (94 lines)

**`smait/sensors/`:**
- Purpose: Low-level data ingestion and preprocessing before ML inference
- Contains: Audio VAD pipeline with ring buffer for raw 4-channel alignment, video frame decoding and buffering
- Key files: `audio_pipeline.py` (280 lines, includes `RawAudioBuffer` and `SpeechSegment`), `video_pipeline.py` (117 lines)

**`smait/perception/`:**
- Purpose: ML model wrappers for audio and visual understanding
- Contains: Face tracking, gaze estimation, engagement detection, speech separation, ASR, transcription filtering, end-of-utterance detection
- Key files: `face_tracker.py` (268 lines), `dolphin_separator.py` (202 lines), `asr.py` (148 lines), `engagement.py` (293 lines), `eou_detector.py` (182 lines)
- This is the largest layer by file count (8 files)

**`smait/dialogue/`:**
- Purpose: LLM-powered conversation management
- Contains: Hybrid local/API LLM with streaming, conversation memory, goodbye detection
- Key files: `manager.py` (312 lines)

**`smait/output/`:**
- Purpose: Text-to-speech synthesis and audio streaming
- Contains: Kokoro-82M TTS with sentence-level streaming pipeline
- Key files: `tts.py` (191 lines)

**`smait/session/`:**
- Purpose: Interaction lifecycle state machine
- Contains: 5-state FSM with proactive greetings, face reacquisition, timeout handling
- Key files: `manager.py` (251 lines)

**`smait/utils/`:**
- Purpose: Cross-cutting utilities for logging and metrics
- Contains: Structured JSON session logger with HRI checklist scoring, latency/performance tracker
- Key files: `data_logger.py` (251 lines), `metrics.py` (123 lines)

**`isaac_sim_tests/`:**
- Purpose: Domain randomization test framework for validating ASD/speaker detection in simulation
- Contains: 5-phase test runner (ASD tuning, AV delay tolerance, noise robustness, articulation variation, multi-speaker), domain randomization helpers, metrics collection
- Key files: `run_test.py` (635 lines), `config.yaml`
- Note: References v2 APIs (LASER ASD, SpeakerVerifier) not present in v3 codebase

**`scripts/`:**
- Purpose: Standalone CLI tools
- Contains: SQLite-based metrics tracker for logging test results and parameter changes over time
- Key files: `track_metrics.py` (294 lines)
- Note: References v2 config fields (min_lip_movement, min_mar_for_speech) not in v3 config

**`docs/`:**
- Purpose: Documentation, hardware specs, integration guides
- Contains: CAE SDK integration guide, Isaac Sim setup, hardware spec images, reference SDK code
- Key files: `CAE_INTEGRATION_GUIDE.md`, `isaac_sim_setup.md`
- `docs/hardware-sdk/` contains Android reference SDK code (CAEDemoAIUI) - large, not part of the Python project

## Key File Locations

**Entry Points:**
- `run_jackie.py`: Primary system entry point (CLI args -> Config -> HRISystem -> asyncio.run)
- `smait/demo.py`: Component-level demo scripts (`python -m smait.demo <component>`)
- `run_tests.py`: Test harness runner (references v2 modules)

**Configuration:**
- `smait/core/config.py`: All configuration dataclasses and loading logic
- `.env.example`: Environment variable template (OPENAI_API_KEY, SMAIT_* vars)
- `pyproject.toml`: Project metadata, dependencies, black/ruff/pytest config
- `requirements.txt`: Pip dependencies with installation notes for non-PyPI packages

**Core Logic:**
- `smait/main.py`: `HRISystem` class - top-level orchestrator (438 lines)
- `smait/core/events.py`: `EventBus` and all `EventType` definitions
- `smait/perception/dolphin_separator.py`: Core innovation - AV-TSE speaker separation
- `smait/dialogue/manager.py`: LLM dialogue with hybrid local/API streaming
- `smait/session/manager.py`: Session lifecycle FSM

**Testing:**
- `isaac_sim_tests/`: Simulation-based testing framework
- `run_tests.py`: Integration test harness (v2 references)
- No `tests/` directory exists despite `pyproject.toml` setting `testpaths = ["tests"]`

## Naming Conventions

**Files:**
- `snake_case.py` for all Python modules: `audio_pipeline.py`, `face_tracker.py`, `data_logger.py`
- `manager.py` for orchestrator/lifecycle modules: `smait/connection/manager.py`, `smait/dialogue/manager.py`, `smait/session/manager.py`
- `__init__.py` for package exports (explicit `__all__` in `connection/`)

**Directories:**
- `snake_case` for all directories: `domain_randomization/`, `isaac_sim_tests/`
- Organized by domain/layer, not by type

**Classes:**
- `PascalCase` for all classes: `HRISystem`, `AudioPipeline`, `FaceTracker`, `DialogueManager`
- Dataclasses for data containers: `SpeechSegment`, `FaceTrack`, `GazeResult`, `SeparationResult`, `TranscriptResult`
- Enums for state/type: `EventType`, `FrameType`, `SessionState`, `EngagementState`

**Functions/Methods:**
- `snake_case` for all functions: `process_frame()`, `init_model()`, `ask_streaming()`
- Private methods prefixed with `_`: `_wire_events()`, `_handle_binary()`, `_check_filters()`
- Async methods use `async def`: `init_model()`, `start()`, `stop()`, `separate()`
- Event handlers prefixed with `_on_`: `_on_tts_start()`, `_on_face_lost()`

**Constants:**
- `UPPER_SNAKE_CASE`: `MAX_LOOP_RETRIES`, `IOU_THRESHOLD`, `HALLUCINATION_PHRASES`, `SENTENCE_BOUNDARY`

## Where to Add New Code

**New Perception Module (e.g., emotion detection, gesture recognition):**
- Implementation: `smait/perception/<module_name>.py`
- Pattern: Class with `__init__(self, config: Config, event_bus: EventBus)` and `async def init_model()`
- Wire into `HRISystem.__init__()` for instantiation and `HRISystem._wire_events()` for event subscriptions
- Add config section: new `@dataclass` in `smait/core/config.py`, add field to `Config`
- Add event types: new entries in `EventType` enum in `smait/core/events.py`

**New Sensor Input (e.g., depth camera, IMU):**
- Implementation: `smait/sensors/<sensor_name>.py`
- Pattern: Class producing dataclass outputs, emitting events via EventBus
- Add binary frame type in `smait/connection/protocol.py::FrameType`
- Add demux case in `smait/connection/manager.py::_handle_binary()`

**New Output Modality (e.g., gesture commands, LED control):**
- Implementation: `smait/output/<modality>.py`
- Pattern: Subscribe to relevant events (e.g., DIALOGUE_RESPONSE), emit output via ConnectionManager
- Add send method to `smait/connection/manager.py`
- Add message schema to `smait/connection/protocol.py::MessageSchema`

**New Dialogue Backend (e.g., Anthropic, Gemini):**
- Add method in `smait/dialogue/manager.py` following `_ask_ollama()` / `_ask_api()` pattern
- Add streaming method following `_stream_ollama()` / `_stream_api()` pattern
- Add config fields to `DialogueConfig` in `smait/core/config.py`

**New Utility:**
- Implementation: `smait/utils/<utility_name>.py`
- Pattern: Standalone class, instantiated in `HRISystem.__init__()`

**New Demo Script:**
- Add async function in `smait/demo.py`, register in `DEMOS` dict at bottom of file

**New Configuration Section:**
- Add `@dataclass` class in `smait/core/config.py` (follow existing pattern like `AudioConfig`)
- Add field to `Config` class with `field(default_factory=NewConfig)`
- Environment variables auto-load with `SMAIT_<SECTION>_<FIELD>` pattern

## Special Directories

**`venv/` and `.venv/`:**
- Purpose: Python virtual environments (both present)
- Generated: Yes
- Committed: No

**`logs/`:**
- Purpose: Session JSON logs and audio WAV files (created at runtime by `DataLogger`)
- Generated: Yes, at `logs/<event-name>/<session-id>.json`
- Committed: No

**`docs/hardware-sdk/`:**
- Purpose: Reference Android SDK code for iFLYTEK CAE integration
- Generated: No (vendor reference code)
- Committed: Yes (large directory tree)

**`.planning/`:**
- Purpose: Planning and analysis documents for development tooling
- Generated: By analysis tools
- Committed: Varies

**`data/`:**
- Purpose: SQLite metrics database (created by `scripts/track_metrics.py`)
- Generated: Yes, at `data/smait_metrics.db`
- Committed: No

---

*Structure analysis: 2026-03-09*
