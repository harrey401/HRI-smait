# Technology Stack

**Analysis Date:** 2026-03-09

## Languages

**Primary:**
- Python 3.10+ - All server-side HRI logic (`smait/` package, 30+ modules)

**Secondary:**
- Kotlin - Android companion app (`smait-jackie-app/app/src/main/java/com/gow/smaitrobot/MainActivity.kt`)
- Java - Android CAE (Computational Audio Engine) SDK integration (`smait-jackie-app/app/src/main/java/com/voice/`)

## Runtime

**Environment:**
- Python 3.10 / 3.11 / 3.12 (configured in `pyproject.toml`)
- CUDA-capable GPU required for ML models (PyTorch)
- uvloop for high-performance async event loop on Linux (`run_jackie.py` lines 29-33)

**Package Manager:**
- pip with setuptools build backend
- Lockfile: Not present (no `requirements.lock` or `pip.lock`)

**Companion App:**
- Android SDK 35 (compileSdk/targetSdk), minSdk 23
- Gradle with Kotlin DSL (`build.gradle.kts`)
- armeabi-v7a NDK filter (robot hardware is ARM)

## Frameworks

**Core:**
- asyncio - Primary concurrency model; entire system is async event-driven
- websockets 12.0+ - WebSocket server for Jackie robot communication (`smait/connection/manager.py`)
- PyTorch 2.0+ - ML model runtime (VAD, Dolphin, L2CS-Net, ASR)

**Testing:**
- pytest 7.0+ with pytest-asyncio 0.21+ - Unit testing (`pyproject.toml` `[tool.pytest.ini_options]`)
- Custom TestHarness - Integration testing with live robot (`smait/utils/test_harness.py`, `run_tests.py`)

**Build/Dev:**
- setuptools 61.0+ with wheel - Package build (`pyproject.toml` build-system)
- black 23.0+ - Code formatting (line-length=100, target py310-py312)
- ruff 0.1.0+ - Linting (rules: E, F, W, I, N, UP; ignores E501)

## Key Dependencies

**Critical (GPU ML Models):**
- `torch>=2.0.0` + `torchaudio>=2.0.0` - PyTorch runtime for all ML models
- `nemo_toolkit[asr]>=2.0.0` - NVIDIA NeMo for Parakeet TDT ASR (`smait/perception/asr.py`)
- `mediapipe>=0.10.0` - Face Mesh 468-landmark face tracking (`smait/perception/face_tracker.py`)
- `transformers>=4.35.0` - HuggingFace transformers for EOU turn detector fallback (`smait/perception/eou_detector.py`)

**Critical (External Installs - not on PyPI):**
- `dolphin` (JusperLee/Dolphin) - Audio-Visual Target Speaker Extraction (`smait/perception/dolphin_separator.py`). Install: `pip install git+https://github.com/JusperLee/Dolphin.git`
- `kokoro` (hexgrad/Kokoro-82M) - TTS engine (`smait/output/tts.py`). Install: `pip install kokoro` or from HuggingFace
- `l2cs` (Ahmednull/L2CS-Net) - Gaze estimation (`smait/perception/gaze.py`). Install: `pip install git+https://github.com/Ahmednull/L2CS-Net.git`
- `livekit.plugins.turn_detector` - End-of-utterance detection (`smait/perception/eou_detector.py`). Install: `pip install git+https://github.com/livekit/turn-detector.git`

**Infrastructure:**
- `opencv-python>=4.8.0` - Video frame decode, image processing (`smait/sensors/video_pipeline.py`, `smait/perception/`)
- `numpy>=1.24.0` - Array operations throughout entire pipeline
- `openai>=1.0.0` - OpenAI API client for LLM fallback (`smait/dialogue/manager.py`)
- `requests>=2.31.0` - HTTP client for Ollama API (`smait/dialogue/manager.py`)
- `websockets>=12.0` - WebSocket server (`smait/connection/manager.py`)
- `sounddevice>=0.4.6` - Audio device access
- `python-dotenv>=1.0.0` - Environment variable loading
- `uvloop>=0.19.0` - Fast event loop (Linux only, optional)
- `psutil` - System monitoring
- `einops` - Tensor operations (Dolphin dependency)

**Android App:**
- `cae.jar` + `AlsaRecorder.jar` - Proprietary robot audio hardware SDK (`smait-jackie-app/app/libs/`)
- `okhttp:4.12.0` - WebSocket client for connecting to Python server
- AndroidX (appcompat, core-ktx, activity, constraintlayout, material)

## Configuration

**Environment:**
- `.env` file present - contains runtime secrets (OPENAI_API_KEY, backend selection)
- `.env.example` documents required variables (`OPENAI_API_KEY`, `SMAIT_ASR_BACKEND`, `SMAIT_ASD_BACKEND`, `CAMERA_INDEX`, `SMAIT_DEBUG`, `SMAIT_SHOW_VIDEO`)
- Environment variables use `SMAIT_<SECTION>_<FIELD>` pattern (e.g., `SMAIT_CONNECTION_PORT=9000`)

**Config System:**
- Dataclass-based hierarchical config: `smait/core/config.py`
- Singleton pattern via `get_config()` / `reset_config()`
- Loading priority: JSON file > environment variables > dataclass defaults
- Sections: `ConnectionConfig`, `AudioConfig`, `SeparationConfig`, `ASRConfig`, `EOUConfig`, `VisionConfig`, `GazeConfig`, `EngagementConfig`, `DialogueConfig`, `TTSConfig`, `SessionConfig`, `LogConfig`

**Build:**
- `pyproject.toml` - Package metadata, dependencies, tool config (black, ruff, pytest)
- `requirements.txt` - Flat dependency list with comments (alternative to pyproject.toml extras)

## Platform Requirements

**Development:**
- Linux (uvloop, ALSA audio via robot hardware)
- NVIDIA GPU with CUDA support (RTX 5070 / Blackwell sm_120 supported with `NEMO_DISABLE_CUDA_GRAPHS=1`)
- Estimated VRAM: ~4-5GB total (Parakeet ~2GB, Kokoro ~1GB, Dolphin ~1GB, L2CS ~0.3GB)
- Ollama running locally at `http://localhost:11434` for local LLM (Phi-4 Mini)
- Python 3.10+

**Production (Robot Deployment):**
- PC/server with NVIDIA GPU running `run_jackie.py`
- Android device (Jackie robot) running companion app, connecting via WebSocket to PC on port 8765
- Robot hardware: 4-channel mic array with CAE processing, camera, speaker
- Network: Wi-Fi between robot (Android) and server (PC), same LAN

**Testing/Simulation:**
- NVIDIA Isaac Sim for simulation testing (`isaac_sim_tests/`)
- NVIDIA Audio2Face for facial animation ground truth (port 12030 streaming, port 8011 HTTP)

## Entry Points

- `run_jackie.py` - Primary server entry point (CLI with argparse)
- `run_tests.py` - Test runner with TestHarness for metrics collection
- `smait.main:run` - Package entry point (via `pyproject.toml` scripts)
- `smait.demo:run` - Demo entry point

---

*Stack analysis: 2026-03-09*
