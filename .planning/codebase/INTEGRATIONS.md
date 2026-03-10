# External Integrations

**Analysis Date:** 2026-03-09

## APIs & External Services

**LLM - Dialogue Generation:**
- **Ollama (local, primary)** - Phi-4 Mini model for low-latency dialogue
  - SDK/Client: `requests` library (HTTP POST to REST API)
  - Endpoint: `http://localhost:11434/api/chat` (streaming and non-streaming)
  - Auth: None (local service)
  - Config: `smait/core/config.py` `DialogueConfig.local_model = "phi-4-mini"`
  - Implementation: `smait/dialogue/manager.py` methods `_ask_ollama()`, `_stream_ollama()`

- **OpenAI API (cloud, fallback)** - GPT-4o-mini for when Ollama fails
  - SDK/Client: `openai` Python SDK (`openai.AsyncOpenAI()`)
  - Auth: `OPENAI_API_KEY` env var
  - Config: `smait/core/config.py` `DialogueConfig.api_model = "gpt-4o-mini"`
  - Implementation: `smait/dialogue/manager.py` methods `_ask_api()`, `_stream_api()`
  - Fallback behavior: tries local Ollama first when `DialogueConfig.try_local_first = True` (default), falls back to OpenAI on timeout/error

## ML Models (Local GPU)

**Voice Activity Detection:**
- **Silero VAD** - Speech detection on CAE audio
  - Loaded via: `torch.hub.load("snakers4/silero-vad", "silero_vad")`
  - Runs on: GPU (via PyTorch)
  - Input: 30ms chunks of 16kHz mono float32 audio
  - Implementation: `smait/sensors/audio_pipeline.py` class `AudioPipeline`

**Automatic Speech Recognition:**
- **NVIDIA Parakeet TDT 0.6B v2** - High-accuracy ASR with word timestamps
  - Loaded via: `nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")`
  - VRAM: ~2GB
  - Input: Clean separated 16kHz mono float32 audio from Dolphin
  - Output: Text, confidence, word timestamps
  - Implementation: `smait/perception/asr.py` class `ParakeetASR`
  - Note: Requires `NEMO_DISABLE_CUDA_GRAPHS=1` for Blackwell GPUs (sm_120)

**Audio-Visual Speech Separation:**
- **Dolphin AV-TSE** (JusperLee/Dolphin) - Target speaker extraction using lip video + multi-channel audio
  - Loaded via: `DolphinModel.from_pretrained("JusperLee/Dolphin")`
  - VRAM: ~1GB
  - Input: 4-channel raw audio tensor + lip ROI video frames
  - Output: Clean separated mono audio + confidence score
  - Implementation: `smait/perception/dolphin_separator.py` class `DolphinSeparator`
  - Fallback: passthrough (mix to mono) when model unavailable

**Text-to-Speech:**
- **Kokoro-82M** (hexgrad/Kokoro-82M) - Ultra-low latency TTS
  - Loaded via: `KokoroTTS()` from `kokoro` package
  - VRAM: ~1GB
  - Output: 24kHz PCM16 mono audio
  - Implementation: `smait/output/tts.py` class `TTSEngine`
  - Fallback: Send text to Android app for Android TTS when Kokoro unavailable

**End-of-Utterance Detection:**
- **LiveKit Turn Detector** - Semantic turn-taking model
  - Loaded via: `livekit.plugins.turn_detector.EOUModel()` or HuggingFace `livekit/turn-detector`
  - Runs on: CPU (no VRAM cost)
  - Input: Transcript text
  - Output: P(end_of_turn) in [0, 1]
  - Implementation: `smait/perception/eou_detector.py` class `EOUDetector`
  - Fallback: Heuristic based on punctuation and word count

**Face Tracking:**
- **MediaPipe Face Mesh** - 468-landmark face detection and tracking
  - Loaded via: `mp.solutions.face_mesh.FaceMesh()`
  - Runs on: CPU/GPU (MediaPipe manages)
  - Input: BGR video frames
  - Output: Face landmarks, bounding boxes, head pose
  - Implementation: `smait/perception/face_tracker.py` class `FaceTracker`

**Gaze Estimation:**
- **L2CS-Net** (Ahmednull/L2CS-Net) - Appearance-based gaze estimation
  - Loaded via: `l2cs.Pipeline(arch="Gaze360")`
  - VRAM: ~0.3GB
  - Input: Face crops from FaceTracker
  - Output: Yaw/pitch degrees, is_looking_at_robot boolean
  - Implementation: `smait/perception/gaze.py` class `GazeEstimator`
  - Fallback: Head pose from MediaPipe landmarks as gaze proxy

## Robot Hardware Interface (WebSocket)

**Jackie Robot (Android App):**
- Protocol: WebSocket (binary frames + JSON text frames)
- Server: `smait/connection/manager.py` class `ConnectionManager`
- Default: `ws://0.0.0.0:8765`
- Single client connection (one robot at a time)
- Max frame size: 4MB
- Heartbeat: ping every 5s, timeout 15s

**Binary Frame Protocol** (`smait/connection/protocol.py`):
| Type Byte | Name | Direction | Content |
|-----------|------|-----------|---------|
| 0x01 | AUDIO_CAE | Jackie -> PC | CAE-processed single-channel audio |
| 0x02 | VIDEO | Jackie -> PC | JPEG video frame |
| 0x03 | AUDIO_RAW | Jackie -> PC | Raw 4-channel mic array audio |
| 0x04 | CONTROL | Reserved | Control frame |
| 0x05 | TTS_AUDIO | PC -> Jackie | PCM16 TTS audio for playback |

**JSON Text Messages** (`smait/connection/protocol.py`):
- Inbound (Jackie -> PC): `doa` (Direction of Arrival angle), `tts_state`, `config`, `cae_status` (AEC/beamforming/noise suppression)
- Outbound (PC -> Jackie): `state` (idle/engaged + listening/thinking/speaking), `transcript` (user/robot text), `tts_control` (start/end mic gating), `response`, `tts` (fallback text-to-speech)

**Android CAE SDK:**
- Proprietary JARs: `cae.jar`, `AlsaRecorder.jar` in `smait-jackie-app/app/libs/`
- Native libraries: JNI via `jniLibs.srcDirs("libs")` in `smait-jackie-app/app/build.gradle.kts`
- Features: Acoustic Echo Cancellation (AEC), beamforming, noise suppression
- Java interfaces: `smait-jackie-app/app/src/main/java/com/voice/caePk/` and `com/voice/osCaeHelper/`

## Data Storage

**Databases:**
- None (no database)

**File Storage:**
- Local filesystem only
- Session logs: JSON files at `logs/<event-name>/<session-id>.json` (`smait/utils/data_logger.py`)
- Audio recordings: WAV files at `logs/<event-name>/<session-id>/` (when `LogConfig.save_audio = True`)
- System log: `smait.log` (append mode, written by `run_jackie.py`)
- Test results: `test_results/` directory (`run_tests.py`)

**Caching:**
- None (models loaded into GPU VRAM at startup, kept resident)

## Authentication & Identity

**Auth Provider:**
- None - No user authentication
- OpenAI API key via `OPENAI_API_KEY` environment variable
- Ollama runs unauthenticated on localhost

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)

**Logs:**
- Python `logging` module with custom format: `HH:MM:SS.mmm [LEVEL] module: message`
- Dual output: stdout + `smait.log` file (`run_jackie.py` `setup_logging()`)
- Per-session structured JSON logs with HRI checklist scoring (`smait/utils/data_logger.py`)
- Per-component performance metrics: rolling avg, p50, p95 latency (`smait/utils/metrics.py`)

## CI/CD & Deployment

**Hosting:**
- Local deployment on PC with NVIDIA GPU connected to robot via LAN
- No cloud hosting

**CI Pipeline:**
- None detected

## Simulation & Testing Environment

**NVIDIA Isaac Sim:**
- Used for automated HRI testing without physical robot (`isaac_sim_tests/`)
- Config: `isaac_sim_tests/config.yaml` (5 test phases: ASD tuning, AV delay, noise, articulation, multi-speaker)
- Resolution: 1280x720 at 30fps

**NVIDIA Audio2Face:**
- Generates ground truth lip sync for simulation testing
- Streaming port: 12030
- HTTP API port: 8011
- Client: `isaac_sim_tests/domain_randomization/audio2face_client.py`
- Features: Audio-to-blendshape generation, blendshape scaling for articulation variation

## Environment Configuration

**Required env vars:**
- `OPENAI_API_KEY` - Required for OpenAI API fallback LLM

**Optional env vars:**
- `SMAIT_ASR_BACKEND` - ASR engine selection (`parakeet_tdt` or `faster_whisper`)
- `SMAIT_ASD_BACKEND` - Speaker detection backend
- `CAMERA_INDEX` - Camera device index
- `SMAIT_DEBUG` - Enable debug logging (`1`/`true`/`yes`)
- `SMAIT_SHOW_VIDEO` - Show video overlay
- `NEMO_DISABLE_CUDA_GRAPHS` - Required `1` for Blackwell GPUs
- Any `SMAIT_<SECTION>_<FIELD>` pattern for config overrides

**Secrets location:**
- `.env` file in project root (gitignored)
- `.env.example` documents required variables

## Webhooks & Callbacks

**Incoming:**
- None (WebSocket is the only inbound interface)

**Outgoing:**
- None (no outbound webhook calls)

## Event Bus (Internal Integration)

All inter-module communication uses an async pub/sub EventBus (`smait/core/events.py`):

| Event | Producer | Consumer(s) |
|-------|----------|-------------|
| SPEECH_DETECTED | ConnectionManager | AudioPipeline |
| SPEECH_SEGMENT | AudioPipeline | main.py (-> DolphinSeparator) |
| SPEECH_SEPARATED | main.py | main.py (-> Transcriber/ASR) |
| TRANSCRIPT_READY | Transcriber | SessionManager |
| END_OF_TURN | EOUDetector | main.py (-> DialogueManager + TTS) |
| FACE_DETECTED/LOST/UPDATED | FaceTracker | SessionManager, LipExtractor |
| GAZE_UPDATE | GazeEstimator | EngagementDetector |
| ENGAGEMENT_START/LOST | EngagementDetector | SessionManager |
| DIALOGUE_RESPONSE/STREAM | DialogueManager | TTSEngine, ConnectionManager |
| TTS_START/END/AUDIO_CHUNK | TTSEngine | ConnectionManager, AudioPipeline (mic gating) |
| SESSION_START/END | SessionManager, DialogueManager | DataLogger, main.py |
| DOA_UPDATE | ConnectionManager | EngagementDetector, DataLogger |
| CAE_STATUS | ConnectionManager | DataLogger |
| CONNECTION_OPEN/CLOSED | ConnectionManager | SessionManager |

---

*Integration audit: 2026-03-09*
