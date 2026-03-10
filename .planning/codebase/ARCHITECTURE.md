# Architecture

**Analysis Date:** 2026-03-09

## Pattern Overview

**Overall:** Event-Driven Pipeline Architecture with Central EventBus

**Key Characteristics:**
- All inter-module communication flows through a central `EventBus` pub/sub system
- Components are organized into layered pipelines: Sensors -> Perception -> Dialogue -> Output
- Single top-level orchestrator (`HRISystem`) wires all components and manages lifecycle
- Async-first design using `asyncio` with resilient loop restarts (max 10 retries per loop)
- Client-server model: Jackie robot (Android) connects via WebSocket to PC (edge server)
- Graceful degradation: every ML model has a fallback if unavailable (e.g., head pose for gaze, passthrough for separation)

## Layers

**Core (`smait/core/`):**
- Purpose: Configuration management and event system shared across all modules
- Location: `smait/core/`
- Contains: `Config` dataclass hierarchy (singleton), `EventBus` pub/sub, `EventType` enum (29 event types)
- Depends on: Nothing (leaf layer)
- Used by: Every other layer

**Connection (`smait/connection/`):**
- Purpose: WebSocket server handling bidirectional communication with Jackie robot
- Location: `smait/connection/`
- Contains: `ConnectionManager` (server lifecycle, frame demuxing), `protocol.py` (binary frame types, JSON message schemas)
- Depends on: Core (Config, EventBus)
- Used by: HRISystem (wiring), all layers indirectly (via EventBus)
- Protocol: Binary frames with type byte prefix (0x01=CAE audio, 0x02=video, 0x03=raw audio, 0x05=TTS audio), JSON text frames for DOA/state/transcripts

**Sensors (`smait/sensors/`):**
- Purpose: Raw data ingestion and low-level processing (VAD, JPEG decode)
- Location: `smait/sensors/`
- Contains: `AudioPipeline` (Silero VAD + ring buffer + mic gating), `VideoPipeline` (JPEG decode + frame buffer)
- Depends on: Core, receives data from Connection layer via EventBus
- Used by: Perception layer consumes their outputs
- Key data types: `SpeechSegment` (CAE + raw audio with timestamps), `VideoFrame` (decoded BGR image)

**Perception (`smait/perception/`):**
- Purpose: ML-powered understanding of audio and video streams
- Location: `smait/perception/`
- Contains:
  - `face_tracker.py` - MediaPipe Face Mesh (468 landmarks), persistent IOU-based track IDs
  - `lip_extractor.py` - Mouth ROI cropping from face landmarks, temporal buffer per face
  - `gaze.py` - L2CS-Net gaze estimation with head pose fallback
  - `engagement.py` - Multi-signal engagement detection (gaze duration + face area + DOA)
  - `dolphin_separator.py` - Dolphin AV-TSE audio-visual target speaker extraction (core innovation)
  - `asr.py` - NVIDIA Parakeet TDT 0.6B ASR wrapper
  - `transcriber.py` - ASR orchestration with hallucination filtering
  - `eou_detector.py` - LiveKit End-of-Utterance model for semantic turn-taking
- Depends on: Core, Sensors (data types)
- Used by: HRISystem wiring, Dialogue layer (via EventBus)

**Dialogue (`smait/dialogue/`):**
- Purpose: LLM-powered conversational response generation
- Location: `smait/dialogue/`
- Contains: `DialogueManager` with hybrid LLM (local Ollama Phi-4 Mini + OpenAI API fallback), streaming response generation, goodbye detection
- Depends on: Core (Config, EventBus)
- Used by: Output layer (TTS consumes streaming responses)

**Output (`smait/output/`):**
- Purpose: Speech synthesis and audio output
- Location: `smait/output/`
- Contains: `TTSEngine` wrapping Kokoro-82M with sentence-level streaming, mic gating coordination
- Depends on: Core (Config, EventBus)
- Used by: ConnectionManager forwards TTS_AUDIO_CHUNK to Jackie

**Session (`smait/session/`):**
- Purpose: Interaction lifecycle state machine
- Location: `smait/session/`
- Contains: `SessionManager` with 5-state FSM (IDLE -> APPROACHING -> ENGAGED -> CONVERSING -> DISENGAGING), proactive greetings, face reacquisition grace periods
- Depends on: Core (Config, EventBus)
- Used by: HRISystem wiring, controls session-level behavior

**Utilities (`smait/utils/`):**
- Purpose: Logging, metrics, test infrastructure
- Location: `smait/utils/`
- Contains: `DataLogger` (structured JSON session logs + WAV saving + HRI checklist scoring), `MetricsTracker` (rolling latency/performance metrics)
- Depends on: Core (Config)
- Used by: HRISystem for metrics and data logging

## Data Flow

**Primary Audio Pipeline (Speech -> Response):**

1. Jackie sends binary audio frames (0x01 CAE or 0x03 raw) over WebSocket
2. `ConnectionManager._handle_binary()` demuxes by type byte, emits `SPEECH_DETECTED`
3. `AudioPipeline.process_cae_audio()` runs Silero VAD on 30ms chunks, accumulates speech
4. On silence after speech, `AudioPipeline._emit_segment()` produces `SpeechSegment`, emits `SPEECH_SEGMENT`
5. `HRISystem._wire_events()::on_speech_segment()` extracts lip frames from `LipExtractor`, runs `DolphinSeparator.separate()`, emits `SPEECH_SEPARATED`
6. `HRISystem._wire_events()::on_speech_separated()` runs `Transcriber.process_separated_audio()` (Parakeet ASR + hallucination filter), updates `EOUDetector`
7. `EOUDetector.on_silence()` evaluates P(end_of_turn), emits `END_OF_TURN` when confident
8. `HRISystem._wire_events()::on_end_of_turn()` calls `DialogueManager.ask_streaming()`, pipes response through `TTSEngine.speak_streaming()`
9. `TTSEngine` synthesizes sentence-by-sentence, emits `TTS_AUDIO_CHUNK` events
10. `ConnectionManager._on_tts_audio_chunk()` sends binary 0x05 frames to Jackie for playback

**Vision Pipeline (Face -> Engagement):**

1. Jackie sends JPEG frames (0x02) over WebSocket
2. `ConnectionManager` emits `FACE_UPDATED` with raw JPEG bytes
3. `VideoPipeline.process_jpeg()` decodes JPEG to BGR numpy array
4. `FaceTracker.process_frame()` runs MediaPipe Face Mesh, maintains persistent track IDs via IOU matching
5. `GazeEstimator.estimate()` runs L2CS-Net (or head pose fallback) per tracked face
6. `LipExtractor.extract()` crops mouth ROI for Dolphin input
7. `EngagementDetector.update()` evaluates gaze duration + face area + walking-past filter
8. On sustained gaze (>2s), emits `ENGAGEMENT_START` -> `SessionManager` starts session with proactive greeting

**Mic Gating (Echo Suppression):**

1. `TTSEngine` emits `TTS_START` before synthesis
2. `AudioPipeline._on_tts_start()` sets `_mic_gated = True`, suppresses VAD output
3. `ConnectionManager._on_tts_start()` sends `tts_control: start` to Jackie (hardware mic gate)
4. After synthesis completes, `TTS_END` reverses both gates

**State Management:**
- Global config: Singleton `Config` dataclass, loaded from JSON file or environment variables (`SMAIT_*` prefix)
- Session state: `SessionManager` FSM tracks interaction lifecycle per user
- Conversation memory: `DialogueManager._history` sliding window (last 10 turns)
- Per-component state: Each component maintains its own internal state (e.g., `FaceTracker._tracks`, `AudioPipeline._speech_buffer`)
- No database: All state is in-memory. Session logs written to JSON files on disk.

## Key Abstractions

**EventBus (`smait/core/events.py`):**
- Purpose: Decoupled pub/sub for all inter-module communication
- Pattern: Observer with support for both sync and async handlers
- 29 event types covering audio, vision, dialogue, TTS, session, and system events
- `emit()` schedules async handlers as tasks, calls sync handlers directly
- `emit_async()` awaits all async handlers (used sparingly)

**Config (`smait/core/config.py`):**
- Purpose: Typed configuration hierarchy with multiple loading strategies
- Pattern: Nested dataclasses with singleton access via `get_config()`
- 12 config sections: connection, audio, separation, asr, eou, vision, gaze, engagement, dialogue, tts, session, logging
- Loading priority: CLI args override file overrides env vars override defaults

**SpeechSegment (`smait/sensors/audio_pipeline.py`):**
- Purpose: Aligned audio bundle for the separation pipeline
- Contains: CAE-processed mono audio + optional raw 4-channel audio + time window
- The raw audio alignment via `RawAudioBuffer` ring buffer is critical for Dolphin's multichannel input

**FaceTrack (`smait/perception/face_tracker.py`):**
- Purpose: Persistent face identity across frames
- Contains: track_id, bbox, 468 MediaPipe landmarks, head pose, confidence, is_target flag
- IOU matching maintains identity across frames; 2-second timeout before declaring face lost

**SessionState (`smait/session/manager.py`):**
- Purpose: Interaction lifecycle FSM
- States: IDLE -> APPROACHING -> ENGAGED -> CONVERSING -> DISENGAGING
- Manages proactive greetings, face loss grace periods (8s), silence timeouts (30s), reacquisition windows (20s)

## Entry Points

**`run_jackie.py` (Primary):**
- Location: `run_jackie.py`
- Triggers: `python run_jackie.py [--host] [--port] [--voice-only] [--debug] [--config]`
- Responsibilities: Parse CLI args, configure logging, load Config, create and run `HRISystem`
- Sets up uvloop for async performance on Linux
- Disables CUDA graphs for Blackwell GPUs (`NEMO_DISABLE_CUDA_GRAPHS=1`)

**`smait/demo.py` (Component Testing):**
- Location: `smait/demo.py`
- Triggers: `python -m smait.demo <component>` (connection, audio, video, asr, tts, dialogue, eou, separation, full)
- Responsibilities: Individual component demo scripts for isolated testing

**`run_tests.py` (Test Runner):**
- Location: `run_tests.py`
- Triggers: `python run_tests.py [--sessions N] [--voice-only] [--wer]`
- Responsibilities: Runs full system with TestHarness for metrics recording
- Note: References `smait.sensors.network_source` and `smait.utils.test_harness` which appear to be from v2 and may not exist in current codebase

**`isaac_sim_tests/run_test.py` (Simulation Testing):**
- Location: `isaac_sim_tests/run_test.py`
- Triggers: `python run_test.py --phase 1 2 3` (or `--all`, `--quick`)
- Responsibilities: Domain randomization testing via Isaac Sim + Audio2Face for ASD validation

## Error Handling

**Strategy:** Graceful degradation with resilient loops and component-level fallbacks

**Patterns:**
- **Resilient loops:** `HRISystem._resilient_loop()` wraps each async loop (audio, video, session_timeout) with auto-restart on exception (max 10 retries, 1s cooldown between retries)
- **Model fallbacks:** Every ML component has a degraded path:
  - Dolphin unavailable -> passthrough CAE audio
  - L2CS-Net unavailable -> head pose estimation
  - Parakeet unavailable -> ASR disabled (transcripts rejected)
  - Kokoro unavailable -> Android TTS fallback via JSON message
  - LiveKit EOU unavailable -> heuristic punctuation-based EOU
  - Ollama unavailable -> OpenAI API; both fail -> graceful error message
- **Event handler safety:** `EventBus._safe_async_call()` catches and logs all handler exceptions without propagating
- **Connection handling:** Single client mode with clean disconnect/reconnect; `ConnectionManager` rejects second connections

## Cross-Cutting Concerns

**Logging:** Python `logging` module throughout. `run_jackie.py::setup_logging()` configures dual handlers (stdout + `smait.log` file). Noisy libraries silenced (websockets, mediapipe, nemo, torch). All components use `logging.getLogger(__name__)`.

**Validation:** `Transcriber._check_filters()` validates ASR output (confidence threshold, hallucination phrase detection). `AudioPipeline._emit_segment()` rejects segments shorter than 0.5s. `ConnectionManager._handle_binary()` validates frame length and type byte.

**Authentication:** None. Single-client WebSocket connection model assumes trusted local network between Jackie and PC.

**Metrics:** `MetricsTracker` provides start/stop timer pairs and rolling averages/percentiles for separation, ASR, and dialogue latency. `DataLogger` produces structured JSON session logs with HRI checklist scoring (0-7 scale).

---

*Architecture analysis: 2026-03-09*
