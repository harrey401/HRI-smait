# Codebase Concerns

**Analysis Date:** 2026-03-09

## Tech Debt

**Version mismatch between `pyproject.toml` and `requirements.txt`:**
- Issue: `pyproject.toml` declares version `2.0.0` and lists `faster-whisper>=1.0.0` as a dependency, but `requirements.txt` references `nemo_toolkit[asr]>=2.0.0` (Parakeet). The actual codebase uses Parakeet, not Faster Whisper. The two dependency manifests are out of sync.
- Files: `pyproject.toml` (line 42), `requirements.txt`
- Impact: Installing via `pip install .` pulls wrong ASR dependency. Contributors get confused about which ASR backend is canonical.
- Fix approach: Remove `faster-whisper` from `pyproject.toml` dependencies. Add `nemo_toolkit[asr]`, `websockets`, `torchaudio`, `einops`, `transformers`, `uvloop`, and `psutil` to align with `requirements.txt`. Consider dropping `requirements.txt` in favor of `pyproject.toml` as single source of truth.

**Placeholder URLs and email in `pyproject.toml`:**
- Issue: Project URLs use `yourusername` placeholder and author email is `your.email@example.com`.
- Files: `pyproject.toml` (lines 13, 70-73)
- Impact: Broken links, unprofessional metadata if published.
- Fix approach: Replace with actual GitHub repository URLs and author email.

**`run_tests.py` references non-existent modules:**
- Issue: `run_tests.py` imports `init_jackie_sources`, `set_jackie_server` from `smait.main`, `TestHarness` from `smait.utils.test_harness`, and `JackieWebSocketServer` from `smait.sensors.network_source`. None of these exist in the current codebase. This file references an older architecture (v2) that has been replaced.
- Files: `run_tests.py` (lines 27-28, 48-51)
- Impact: Test runner is completely non-functional. Cannot run structured test sessions.
- Fix approach: Rewrite `run_tests.py` to use the current `HRISystem` API and `ConnectionManager`. Create `smait/utils/test_harness.py` if test harness functionality is needed.

**`.env` file committed to repository (not in `.gitignore` initially):**
- Issue: `.env` file exists and is 551 bytes. While `.gitignore` includes `.env`, the file is listed as untracked in git status (new repo), meaning it could be committed accidentally. The `.env.example` references `SMAIT_ASD_BACKEND=laser` which is a v2 concept not present in v3.
- Files: `.env`, `.env.example`
- Impact: Secrets exposure risk. Stale `.env.example` misleads developers.
- Fix approach: Ensure `.env` is never committed. Update `.env.example` to reflect v3 configuration variables (`OPENAI_API_KEY`, `SMAIT_CONNECTION_PORT`, `SMAIT_DIALOGUE_API_MODEL`, etc.).

**Deprecated `asyncio.get_event_loop()` usage:**
- Issue: `DialogueManager` uses `asyncio.get_event_loop().run_in_executor()` in two places. This is deprecated in Python 3.10+ and will emit warnings. The correct approach is `asyncio.get_running_loop().run_in_executor()`.
- Files: `smait/dialogue/manager.py` (lines 162, 220)
- Impact: Deprecation warnings in logs. Will break in future Python versions.
- Fix approach: Replace `asyncio.get_event_loop()` with `asyncio.get_running_loop()` in both `_ask_ollama()` and `_stream_ollama()`.

**Ollama streaming uses blocking `requests` in async context:**
- Issue: `_stream_ollama()` runs `requests.post(..., stream=True)` in an executor, then iterates `resp.iter_lines()` synchronously in the async generator. The initial POST is in an executor, but `iter_lines()` blocks the event loop.
- Files: `smait/dialogue/manager.py` (lines 217-249)
- Impact: During Ollama streaming, the event loop is blocked for each line iteration, stalling audio processing and other async tasks. This degrades real-time responsiveness.
- Fix approach: Use `httpx.AsyncClient` or `aiohttp` for truly async streaming. Alternatively, run the entire iteration in a thread and use a `asyncio.Queue` to yield chunks.

**Global mutable singleton for Config:**
- Issue: `_config_instance` is a module-level global mutable singleton with `get_config()` and `reset_config()`. CLI args directly mutate config fields after creation (`config.connection.host = args.host`).
- Files: `smait/core/config.py` (lines 14, 205-219), `run_jackie.py` (lines 97-100)
- Impact: Config state leaks between tests. Direct mutation makes it hard to track what changed.
- Fix approach: Pass config explicitly rather than relying on singleton. Use a builder pattern or `dataclasses.replace()` for creating modified configs.

## Known Bugs

**`_wire_events` accesses private `_current_turn` on DataLogger:**
- Symptoms: `on_end_of_turn` handler in `smait/main.py` directly accesses `self.data_logger._current_turn` to set `robot_text` and `llm_latency_ms`. This is fragile and violates encapsulation.
- Files: `smait/main.py` (lines 310-312)
- Trigger: Every dialogue turn.
- Workaround: Works but is brittle. If `DataLogger` internals change, this silently breaks.

**EventBus caches first event loop permanently:**
- Symptoms: `EventBus._loop` is set on the first `emit()` call and never updated. If the event loop changes (e.g., during testing with `asyncio.run()`), stale loop reference causes `RuntimeError: Event loop is closed`.
- Files: `smait/core/events.py` (lines 92-96)
- Trigger: Running multiple `asyncio.run()` calls in tests or demos.
- Workaround: Call `asyncio.get_running_loop()` on every `emit()` instead of caching.

**Video pipeline `process_jpeg` never called from event system:**
- Symptoms: `VideoPipeline.process_jpeg()` is never subscribed to any event. In `_video_loop()` (main.py line 408), the code reads `self.video_pipeline.latest_frame`, but nothing populates it because the FACE_UPDATED event from `ConnectionManager._handle_binary()` emits raw JPEG bytes, and there is no handler that calls `video_pipeline.process_jpeg()`.
- Files: `smait/main.py` (lines 395-431), `smait/connection/manager.py` (lines 132-137), `smait/sensors/video_pipeline.py`
- Trigger: Any video frame from Jackie.
- Workaround: The video pipeline is effectively non-functional. A handler must be wired to decode JPEG frames from CONNECTION events and feed them to `VideoPipeline.process_jpeg()`.

**Session state notification uses wrong event type:**
- Symptoms: `SessionManager._transition()` emits `EventType.CONNECTION_OPEN` with a `{"action": "send_state", ...}` payload to notify Jackie of state changes. But `CONNECTION_OPEN` is a connection lifecycle event, not a command channel. No handler processes this payload.
- Files: `smait/session/manager.py` (lines 226-230)
- Trigger: Every session state transition.
- Workaround: State changes are never sent to Jackie via this path. The `on_end_of_turn` handler in `main.py` separately calls `connection.send_state()`, partially masking this bug.

## Security Considerations

**WebSocket server binds to 0.0.0.0 with no authentication:**
- Risk: Any device on the network can connect to the WebSocket server. There is no authentication, API key, or TLS. A malicious actor could inject audio/video frames, send fake DOA data, or intercept robot responses.
- Files: `smait/connection/manager.py` (lines 52-58), `smait/core/config.py` (line 19)
- Current mitigation: Single-client limit (second connection rejected with code 1008). Network isolation assumed.
- Recommendations: Add a shared secret or token-based auth in the WebSocket handshake. Enable TLS (wss://) for production. Add origin validation.

**No input validation on incoming JSON messages:**
- Risk: `MessageSchema.parse_text_message()` simply calls `json.loads()` with no schema validation. Malformed or malicious payloads could cause unexpected behavior.
- Files: `smait/connection/protocol.py` (lines 71-73), `smait/connection/manager.py` (lines 145-171)
- Current mitigation: Individual handlers check `isinstance(data, dict)` and use `.get()` with defaults.
- Recommendations: Add JSON schema validation (e.g., pydantic models or jsonschema) for incoming messages. Validate field types and ranges.

**No rate limiting on WebSocket messages:**
- Risk: A misbehaving or compromised client could flood the server with messages, causing memory exhaustion or CPU overload.
- Files: `smait/connection/manager.py`
- Current mitigation: `max_size=2**22` (4MB) frame size limit.
- Recommendations: Add per-second message rate limits. Implement backpressure when processing queues grow.

**OpenAI API key potentially in `.env` file:**
- Risk: `.env` file exists with 551 bytes of content. If committed, API keys would be exposed.
- Files: `.env`
- Current mitigation: `.gitignore` includes `.env`.
- Recommendations: Verify `.env` is not tracked. Add pre-commit hook to detect secrets.

## Performance Bottlenecks

**Synchronous ML inference blocks event loop:**
- Problem: `ParakeetASR.transcribe()`, `FaceTracker.process_frame()`, and `GazeEstimator.estimate()` are synchronous CPU/GPU-bound operations called from async handlers without `run_in_executor()`.
- Files: `smait/perception/asr.py` (line 74, `transcribe` is sync), `smait/perception/face_tracker.py` (line 67, `process_frame` is sync), `smait/perception/gaze.py` (line 70, `estimate` is sync)
- Cause: These are called from async event handlers or async loops but perform heavy computation synchronously. MediaPipe face mesh processing is ~10-30ms per frame, Parakeet inference is 50-200ms.
- Improvement path: Wrap compute-heavy calls in `asyncio.get_running_loop().run_in_executor()` using a `ThreadPoolExecutor`. Alternatively, use a dedicated processing thread with an asyncio Queue.

**Video loop polls at 30 FPS regardless of frame availability:**
- Problem: `_video_loop()` in `main.py` polls `video_pipeline.latest_frame` every 33ms in a busy loop, even when no new frames exist. This wastes CPU cycles and creates latency (up to 33ms before a new frame is processed).
- Files: `smait/main.py` (lines 395-431)
- Cause: Polling-based design instead of event-driven.
- Improvement path: Use an `asyncio.Event` or `asyncio.Queue` to signal when a new frame is available. Process immediately upon arrival instead of polling.

**Audio WAV files saved on every speech segment:**
- Problem: `DataLogger.save_audio_wav()` is called twice per speech segment (CAE and separated audio). Each write involves disk I/O in the event handler path.
- Files: `smait/main.py` (lines 229-236)
- Cause: Debug audio saving is always enabled when `config.logging.save_audio` is True.
- Improvement path: Offload WAV saving to a background thread/queue. Consider making it configurable per-segment (only save first N or on error).

**GPU VRAM pressure from multiple models:**
- Problem: The system loads 4+ GPU models simultaneously: Silero VAD (~50MB), Dolphin AV-TSE (~2GB), Parakeet ASR (~2GB), Kokoro TTS (~1GB), L2CS-Net (~300MB). Total ~5.3GB VRAM minimum.
- Files: `smait/main.py` (lines 106-117)
- Cause: All models loaded eagerly at startup.
- Improvement path: Implement lazy loading (load on first use). Consider model offloading or sharing GPU memory. Document minimum VRAM requirements (8GB+).

## Fragile Areas

**Event wiring in `HRISystem._wire_events()`:**
- Files: `smait/main.py` (lines 187-353)
- Why fragile: All inter-module communication is wired in a single 166-line method with deeply nested closures that capture `self`. No type safety on event payloads (all `object`). Adding a new event handler requires modifying this one method. Import statements inside closures (`from smait.sensors.audio_pipeline import SpeechSegment`).
- Safe modification: Add new handlers at the end of the method. Always check `isinstance()` on event data. Test with both voice-only and full modes.
- Test coverage: None.

**Session state machine transitions:**
- Files: `smait/session/manager.py` (lines 197-230)
- Why fragile: State transitions have side effects (emit events, cleanup). Multiple event handlers can trigger conflicting transitions simultaneously (e.g., `ENGAGEMENT_LOST` and `SESSION_END` arriving at the same time). No locking or transition validation.
- Safe modification: Add a state transition table/guard. Ensure only valid transitions are allowed. Add `asyncio.Lock` around state changes.
- Test coverage: None.

**Engagement detection with multiple concurrent faces:**
- Files: `smait/perception/engagement.py` (lines 66-293)
- Why fragile: Per-face state tracking via dictionaries (`_gaze_start`, `_area_history`) can grow unbounded if track IDs keep increasing. The `_select_primary_user` DOA alignment bonus is always 1.2x regardless of actual alignment angle.
- Safe modification: Test with 5+ simultaneous faces. Verify cleanup happens on face loss.
- Test coverage: None.

## Scaling Limits

**Single WebSocket client:**
- Current capacity: 1 connected robot (Jackie).
- Limit: Second connection is rejected immediately. No queuing or multi-robot support.
- Scaling path: Refactor `ConnectionManager` to support multiple clients with per-client state. Use client IDs for routing.

**In-memory conversation history:**
- Current capacity: Last 10 turns (20 messages) in `DialogueManager._history`.
- Limit: History is lost on restart. No persistence across sessions.
- Scaling path: Add conversation persistence (SQLite or file-based). Implement summarization for long conversations.

**Log files grow unbounded:**
- Current capacity: WAV files and JSON logs accumulate in `logs/<event-name>/`.
- Limit: Disk fills up during extended deployments. No rotation or cleanup.
- Scaling path: Add log rotation. Implement automatic cleanup of sessions older than N days. Add disk space monitoring.

## Dependencies at Risk

**Multiple git-installable dependencies without pinned versions:**
- Risk: `dolphin`, `l2cs`, `livekit/turn-detector`, and `kokoro` are all installed from git repositories or HuggingFace without pinned commits. API changes in these libraries will silently break the system.
- Impact: Any of these models failing to load causes a silent fallback to degraded mode (passthrough, heuristics, Android TTS).
- Migration plan: Pin specific git commit hashes. Add version checks at import time. Create a `requirements-lock.txt` with exact versions.

**NeMo 2.0 API instability:**
- Risk: NeMo 2.0 has breaking API changes (tuple return from `transcribe()`). The code has workarounds for this (lines 96-105 in `smait/perception/asr.py`) but future NeMo updates may break again.
- Impact: ASR completely fails if API changes.
- Migration plan: Pin NeMo version. Add integration test that verifies `transcribe()` return format.

## Missing Critical Features

**No automated test suite:**
- Problem: There are zero unit tests, integration tests, or end-to-end tests. The `pyproject.toml` references `testpaths = ["tests"]` but no `tests/` directory exists. `run_tests.py` references modules that do not exist.
- Blocks: Cannot verify correctness after refactoring. No CI/CD possible. No regression detection.

**No health monitoring or watchdog:**
- Problem: If a GPU model silently fails or hangs, there is no health check endpoint or watchdog to detect and recover. The resilient loop only catches exceptions, not hangs.
- Blocks: Unattended deployment at conferences/events where manual restart is impractical.

**No configuration validation at startup:**
- Problem: Config loads with silent defaults. If `OPENAI_API_KEY` is missing, it only fails at first API call. No startup validation that required services (Ollama, GPU) are available.
- Blocks: Delayed failure discovery. System appears to start successfully but fails on first interaction.

**No graceful degradation UI feedback:**
- Problem: When models fail to load, the system silently falls back (e.g., head-pose instead of L2CS gaze, heuristic EOU instead of model). Jackie's Android app has no way to know which capabilities are actually active.
- Blocks: Operators cannot diagnose degraded performance during live demos.

## Test Coverage Gaps

**All modules at 0% test coverage:**
- What's not tested: Every module in `smait/` has zero automated tests. Critical untested areas include:
  - `smait/core/events.py` - EventBus pub/sub, async handler execution
  - `smait/connection/manager.py` - WebSocket message handling, binary frame parsing
  - `smait/sensors/audio_pipeline.py` - VAD pipeline, speech segment production, ring buffer
  - `smait/perception/transcriber.py` - ASR hallucination filtering
  - `smait/perception/eou_detector.py` - End-of-utterance prediction logic
  - `smait/dialogue/manager.py` - LLM fallback chain, streaming, goodbye detection
  - `smait/session/manager.py` - State machine transitions, timeout logic
- Files: All files in `smait/`
- Risk: Any change could break the system silently. The complex event wiring in `main.py` is especially risky since it connects all modules.
- Priority: High. Start with unit tests for `EventBus`, `BinaryFrame`/`MessageSchema`, `AudioPipeline` (ring buffer), `Transcriber` (filters), `EOUDetector` (heuristics), and `SessionManager` (state machine).

---

*Concerns audit: 2026-03-09*
