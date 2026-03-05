# SMAIT HRI v3.0 — Clean Rebuild Prompt

> **Purpose:** One-shot prompt for Claude Code to rebuild the entire SMAIT system from scratch.
> **Date:** 2026-03-04
> **Author:** Gow + Gowfy

---

## 🎯 What You're Building

**SMAIT** (Social Multi-modal AI Interaction Technology) — a real-time Human-Robot Interaction system for **Jackie**, a mobile conference robot. The system enables Jackie to:

1. **Detect** when someone approaches and wants to interact (engagement detection)
2. **Verify** that the person in front of Jackie is the one speaking (active speaker detection)
3. **Transcribe** their speech in real-time (ASR)
4. **Generate** a conversational response (LLM)
5. **Speak** the response back naturally (TTS)
6. **Manage** the full interaction lifecycle (session management)

**Context:** This is a masters project being presented at the HFES (Human Factors and Ergonomics Society) conference in mid-April 2026. The code must be clean, well-documented, and defensible — not a demo hack.

**Environment:** Noisy conference floor with multiple people walking around. The robot must correctly identify WHO is talking to it (not a bystander) and handle interruptions, walk-aways, and re-engagement gracefully.

---

## 🔧 Hardware

### Jackie Robot (Android)
- **SoC:** Rockchip RK3588 (ARM, Android 12)
- **Microphone:** 4-mic circular array with **iFLYTEK CAE SDK**
  - Hardware beamforming (steers toward speaker)
  - Hardware AEC (acoustic echo cancellation — prevents robot hearing its own TTS)
  - Hardware noise suppression
  - DOA (Direction of Arrival) — angle in degrees of detected sound source
- **Camera:** RGB camera (front-facing)
- **Display:** 15.6" touchscreen
- **Speaker:** Built-in speaker for TTS playback
- **Android App:** `smait-jackie-app` — captures audio/video, streams to PC, plays TTS audio, shows UI states

### Lab PC (AI Pipeline)
- **GPU:** NVIDIA RTX 5070 (12GB VRAM, Blackwell architecture)
- **OS:** Ubuntu Linux, Python 3.10 (conda env: "smait")
- **Role:** Runs ALL AI inference (ASR, ASD, LLM, TTS, face tracking, engagement detection)

### Communication: Jackie ↔ PC
- **Primary:** WebSocket over dedicated WiFi network (GL.iNet travel router — isolated 5GHz)
- **Fallback:** USB tethering via ADB port forwarding (`adb forward tcp:8765 tcp:8765`)
- **Protocol:** Binary frames for audio/video, JSON text frames for control messages

---

## 📊 VRAM Budget (RTX 5070 — 12GB)

| Component | Model | VRAM |
|-----------|-------|------|
| ASR | Parakeet TDT 1.1B | ~4.0 GB |
| TTS | Kokoro-82M | ~1.0 GB |
| LLM | Phi-4 Mini 3.8B (Q4) | ~3.0 GB |
| ASD | TalkNet | ~1.5 GB |
| Gaze | L2CS-Net | ~0.3 GB |
| Face/Pose | MediaPipe | ~0.2 GB |
| **Total** | | **~10.0 GB** |
| **Headroom** | | **~2.0 GB** |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Jackie Robot (Android)                  │
│                                                            │
│  Mic Array → iFLYTEK CAE SDK → processed audio            │
│  Camera → RGB frames                                       │
│  Speaker ← TTS audio playback                             │
│  Display ← UI state updates                               │
│                                                            │
│         ↕ WebSocket (ws://router-ip:8765)                 │
└──────────────────────────────────────────────────────────┘
                          ↕
┌──────────────────────────────────────────────────────────┐
│                    Lab PC (Python 3.10)                    │
│                                                            │
│  ConnectionManager (WebSocket + USB fallback)              │
│       ↓                                                    │
│  ┌─ Audio Path ───────────────────────────────────┐       │
│  │ AudioPipeline (Silero VAD) → SpeechSegments    │       │
│  │     → Transcriber (Parakeet TDT 1.1B)          │       │
│  │         → EOU Detector (LiveKit) → turn decision│       │
│  └────────────────────────┬───────────────────────┘       │
│                           ↓                                │
│  ┌─ Speaker Verification ─────────────────────────┐       │
│  │ ASD visual pre-filter (LASER/MediaPipe)         │       │
│  │     → TalkNet audio-visual fusion               │       │
│  │         → DOA spatial check (from iFLYTEK)      │       │
│  │             → ACCEPT / REJECT                   │       │
│  └────────────────────────┬───────────────────────┘       │
│                           ↓ ACCEPT                         │
│  ┌─ Dialogue ─────────────────────────────────────┐       │
│  │ LLM (Phi-4 Mini local / GPT-4o-mini API)       │       │
│  │     → response text                             │       │
│  └────────────────────────┬───────────────────────┘       │
│                           ↓                                │
│  ┌─ Output ───────────────────────────────────────┐       │
│  │ TTS (Kokoro-82M) → audio chunks                 │       │
│  │     → stream to Jackie speaker                  │       │
│  │     → gate mic during playback (echo suppress)  │       │
│  └────────────────────────────────────────────────┘       │
│                                                            │
│  ┌─ Vision Loop (parallel) ───────────────────────┐       │
│  │ Camera frames → MediaPipe Face Mesh             │       │
│  │     → FaceTracker (multi-face, persistent IDs)  │       │
│  │     → L2CS-Net gaze estimation                  │       │
│  │     → EngagementDetector (gaze + distance)      │       │
│  │     → ASD scoring (mouth movement buffer)       │       │
│  └────────────────────────────────────────────────┘       │
│                                                            │
│  ┌─ Session Manager ──────────────────────────────┐       │
│  │ State: IDLE → APPROACHING → ENGAGED → CONV     │       │
│  │        → DISENGAGING → IDLE                     │       │
│  │ Proactive greeting on engagement                │       │
│  │ Farewell on timeout/walk-away/goodbye           │       │
│  └────────────────────────────────────────────────┘       │
│                                                            │
│  ┌─ Data Logger ──────────────────────────────────┐       │
│  │ JSON session logs with per-turn metrics         │       │
│  │ ASR confidence, ASD score, latency breakdown    │       │
│  │ HRI success checklist (auto-scored)             │       │
│  └────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 Package Structure (build this)

```
smait/
├── core/
│   ├── __init__.py
│   ├── config.py          — Dataclass config hierarchy, env/file loading
│   └── events.py          — Async EventBus (pub/sub for all inter-module comms)
├── connection/
│   ├── __init__.py
│   ├── manager.py         — ConnectionManager: WebSocket + USB fallback, auto-reconnect
│   └── protocol.py        — Message types: audio binary, video binary, JSON control
├── sensors/
│   ├── __init__.py
│   ├── audio_pipeline.py  — Silero VAD + ring buffer → SpeechSegment generator
│   └── video_pipeline.py  — Frame capture from WebSocket → frame buffer
├── perception/
│   ├── __init__.py
│   ├── asr.py             — Parakeet TDT 1.1B wrapper (NeMo)
│   ├── transcriber.py     — ASR orchestration: segment → transcript + confidence filter
│   ├── eou_detector.py    — LiveKit End-of-Utterance model (text-based turn prediction)
│   ├── face_tracker.py    — MediaPipe Face Mesh + persistent track IDs
│   ├── gaze.py            — L2CS-Net gaze estimation
│   ├── asd.py             — ASD pipeline: visual pre-filter → TalkNet AV fusion
│   ├── engagement.py      — Engagement detector (gaze + face area distance proxy)
│   └── verifier.py        — SpeakerVerifier state machine (ASD + DOA + session)
├── dialogue/
│   ├── __init__.py
│   └── manager.py         — LLM dialogue (Phi-4 local + API fallback), conversation memory
├── output/
│   ├── __init__.py
│   └── tts.py             — Kokoro-82M TTS, sentence-level streaming, mic gating
├── session/
│   ├── __init__.py
│   └── manager.py         — Session lifecycle state machine, proactive behaviors
├── utils/
│   ├── __init__.py
│   ├── data_logger.py     — Structured JSON interaction logger
│   └── metrics.py         — Performance metrics capture
├── __init__.py
├── main.py                — HRISystem: wires everything, runs async loops
└── demo.py                — Individual component demo scripts
```

---

## 🧩 Module Specifications

### 1. `core/config.py` — Configuration

Dataclass hierarchy. Load from env vars or JSON file. Singleton via `get_config()`.

```python
@dataclass
class Config:
    audio: AudioConfig        # sample_rate=16000, vad_threshold=0.5, silence_duration_ms=300
    asr: ASRConfig            # model="nvidia/parakeet-tdt-1.1b"
    vision: VisionConfig      # max_faces=5, asd_threshold=0.5
    dialogue: DialogueConfig  # local_model="phi-4-mini", api_model="gpt-4o-mini"
    tts: TTSConfig            # engine="kokoro", model="kokoro-82m"
    session: SessionConfig    # timeout=30s, face_lost_grace=8s, reacquisition=20s
    connection: ConnectionConfig  # primary_uri, fallback_uris, reconnect_interval
    engagement: EngagementConfig  # min_gaze_duration=2.0s, proximity_threshold (face area)
    logging: LogConfig        # output_dir, save_audio, save_video_snapshots
    debug: bool = False
    show_video: bool = True
```

### 2. `core/events.py` — Event Bus

Async pub/sub. ALL inter-module communication goes through this. No direct method calls between modules.

**Event types:**
```python
class EventType(Enum):
    # Audio
    SPEECH_DETECTED = auto()
    SPEECH_ENDED = auto()
    
    # ASR
    TRANSCRIPT_READY = auto()      # TranscriptResult(text, confidence, latency_ms, start_time, end_time)
    TRANSCRIPT_REJECTED = auto()   # reason: low_confidence, hallucination
    
    # Turn-taking
    END_OF_TURN = auto()           # EOU detector says user is done
    
    # Vision
    FACE_DETECTED = auto()
    FACE_LOST = auto()
    GAZE_UPDATE = auto()           # GazeResult(yaw, pitch, is_looking_at_robot)
    
    # ASD
    SPEAKER_VERIFIED = auto()      # VerifyResult.ACCEPT
    SPEAKER_REJECTED = auto()      # VerifyResult.REJECT (wrong speaker, no face)
    
    # Engagement
    ENGAGEMENT_START = auto()      # Person approaching + looking at robot
    ENGAGEMENT_LOST = auto()       # Person walked away or stopped looking
    
    # Dialogue
    DIALOGUE_RESPONSE = auto()     # LLM response ready
    
    # TTS
    TTS_START = auto()             # Gate mic
    TTS_END = auto()               # Ungate mic
    
    # Session
    SESSION_START = auto()
    SESSION_END = auto()           # reason: goodbye, timeout, walk_away
    
    # Hardware
    DOA_UPDATE = auto()            # Direction of arrival angle from CAE
    CAE_STATUS = auto()            # Hardware filter states
    CONNECTION_STATE = auto()      # connected, reconnecting, fallback
    
    # System
    ERROR = auto()
```

### 3. `connection/manager.py` — ConnectionManager

Manages WebSocket connection to Jackie with automatic fallback.

```python
class ConnectionManager:
    """
    Tries connection URIs in order:
    1. ws://<travel-router-ip>:8765  (dedicated WiFi)
    2. ws://localhost:8765           (ADB USB forward)
    3. ws://<hotspot-ip>:8765        (phone hotspot)
    
    Features:
    - Auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s)
    - Heartbeat ping/pong every 5s to detect dead connections
    - Emits CONNECTION_STATE events
    - Binary frames: 0x01=audio, 0x02=video, 0x03=control
    - Inbound: audio bytes, video JPEG frames, CAE status JSON, DOA angles
    - Outbound: TTS audio bytes, UI state JSON, text messages
    """
```

### 4. `sensors/audio_pipeline.py` — Audio + VAD

```python
class AudioPipeline:
    """
    Receives audio from ConnectionManager.
    Runs Silero VAD on each chunk (30ms).
    Accumulates speech in ring buffer.
    Emits SpeechSegment when silence > min_silence (300ms initial check).
    
    Key: Does NOT decide end-of-turn alone. It detects speech boundaries,
    then EOU Detector makes the turn decision.
    
    SpeechSegment = {audio: np.ndarray, start_time: float, end_time: float, duration: float}
    """
```

### 5. `perception/asr.py` — Parakeet TDT

```python
class ParakeetASR:
    """
    NVIDIA NeMo Parakeet TDT 1.1B.
    - Load model once at init (GPU)
    - transcribe(audio: np.ndarray) → TranscriptResult(text, confidence, word_timestamps, latency_ms)
    - Handles NeMo 2.0 quirks (tuple return format)
    - Confidence: normalized 0-1 from NeMo logprobs
    """
```

### 6. `perception/eou_detector.py` — End-of-Utterance

```python
class EOUDetector:
    """
    LiveKit's open-weight end-of-utterance model.
    Runs on CPU. Input: current transcript text. Output: P(end_of_turn).
    
    Usage:
    - AudioPipeline detects silence > 300ms
    - Feed transcript so far to EOUDetector
    - If P(eou) > 0.7 → emit END_OF_TURN event → trigger LLM
    - If P(eou) < 0.7 but silence > 1500ms → emit END_OF_TURN anyway (hard cutoff)
    
    This reduces perceived latency from 1000ms to ~300-500ms for clear questions
    while preventing premature cutoffs during thinking pauses.
    """
```

### 7. `perception/face_tracker.py` — Face Tracking

```python
class FaceTracker:
    """
    MediaPipe Face Mesh (468 landmarks per face).
    - Persistent track IDs across frames (IOU/centroid matching)
    - FaceTrack: track_id, bbox, landmarks, head_yaw, head_pitch, last_seen, confidence
    - Handles: new face, face exit, ID re-association after brief occlusion
    - Emits FACE_DETECTED, FACE_LOST events
    - Max 5 tracked faces
    """
```

### 8. `perception/gaze.py` — Gaze Estimation

```python
class GazeEstimator:
    """
    L2CS-Net (~0.3GB VRAM).
    Input: face crop from FaceTracker.
    Output: GazeResult(yaw_deg, pitch_deg, is_looking_at_robot: bool)
    
    is_looking_at_robot = |yaw| < 30° AND |pitch| < 20°
    Emits GAZE_UPDATE events per tracked face.
    """
```

### 9. `perception/asd.py` — Active Speaker Detection

```python
class ActiveSpeakerDetector:
    """
    Three-stage ASD pipeline:
    
    Stage 1 — Visual pre-filter (fast, CPU):
      MediaPipe MAR (Mouth Aspect Ratio) + lip movement delta.
      Quickly identifies faces with mouth movement → candidate speakers.
    
    Stage 2 — TalkNet audio-visual fusion (GPU, ~1.5GB):
      For each candidate, cross-reference audio signal with lip movement video.
      Returns ASD confidence score per face.
    
    Stage 3 — DOA spatial check:
      If iFLYTEK DOA angle available, check if highest-ASD-score face
      is roughly in the DOA direction. Use as tiebreaker / confidence boost.
    
    Maintains a circular buffer of ASD scores (5s, timestamped) for
    temporal lookup when transcript arrives with start/end times.
    
    ASDResult = {face_id, is_speaking: bool, confidence: float, method: str}
    """
```

### 10. `perception/verifier.py` — Speaker Verification State Machine

```python
class SpeakerVerifier:
    """
    Combines ASD + session state to decide: should this transcript be processed?
    
    States: IDLE, ENGAGING, ENGAGED, DISENGAGING
    
    verify_speech(transcript, faces, asd_results, doa_angle) → VerifyOutput:
      - IDLE + engagement detected → start session, ACCEPT
      - ENGAGED + ASD confirms target user → ACCEPT
      - ENGAGED + ASD says different person → REJECT (log: "different_speaker")
      - ENGAGED + no face visible → hold (NO_FACE), wait for re-acquisition
      - Multiple speakers → pick highest ASD + DOA alignment → accept if it's target
    
    VerifyOutput = {result: ACCEPT/REJECT/NO_FACE, text, confidence, reason, face_id, asd_score}
    """
```

### 11. `perception/engagement.py` — Engagement Detection

```python
class EngagementDetector:
    """
    Determines if someone wants to interact. Two signals:
    
    1. Distance proxy: face bounding box area (larger = closer).
       Threshold: configurable face_area_threshold.
    
    2. Gaze: L2CS-Net output. is_looking_at_robot must be True.
    
    Engagement triggered when: both conditions met for > 2 seconds (debounced).
    
    Session lifecycle states:
      IDLE → face detected, growing area + gaze toward robot
        → APPROACHING → sustained gaze > 2s + area stable
          → ENGAGED → proactive greeting triggered
            → CONVERSING → active dialogue
              → DISENGAGING → gaze away > 3s OR area shrinking OR silence > timeout
                → IDLE → cleanup
    
    Rules:
    - Don't greet people walking past (face area changing rapidly)
    - Don't greet groups — wait for individual direct approach
    - Primary user = largest face + direct gaze + DOA alignment
    """
```

### 12. `dialogue/manager.py` — LLM Dialogue

```python
class DialogueManager:
    """
    Hybrid: Phi-4 Mini (3.8B, Q4) local via Ollama + GPT-4o-mini API fallback.
    
    - try_local_first: bool = True. Try Ollama, fall back to API on timeout/error.
    - Maintains conversation history (sliding window, max 10 turns)
    - System prompt: Jackie persona (warm, concise 1-3 sentences, owns being a robot)
    - Goodbye detection: if response indicates farewell → emit SESSION_END
    - ask_async(text) → DialogueResponse(text, latency_ms, model_used, tokens_used)
    
    Streaming: If LLM supports streaming, yield partial responses for
    sentence-level TTS (generate audio for sentence 1 while LLM produces sentence 2).
    """
```

**System prompt:**
```
You are Jackie, a friendly AI-powered conference robot at SJSU. Keep responses to 1-3 spoken
sentences. Be warm, natural, slightly playful. You ARE a robot — own it with personality.
You can see faces, detect who's speaking, and hold real conversations in real-time.
If speech seems garbled: "Sorry, it's noisy — could you say that again?"
Plain spoken sentences only — no lists, markdown, or formatting.
```

### 13. `output/tts.py` — Text-to-Speech

```python
class TTSEngine:
    """
    Kokoro-82M: <100ms TTFB, ~1GB VRAM, 96x real-time.
    
    - speak(text) → audio bytes (wav/pcm)
    - speak_streaming(text_generator) → yields audio chunks per sentence
    - Emits TTS_START (gate mic) and TTS_END (ungate mic)
    - Send audio bytes to Jackie via ConnectionManager (binary frame)
    - Jackie app plays audio through speaker
    
    Sentence-level streaming pipeline:
      LLM streams → detect sentence boundary (.!?) → Kokoro generates that sentence
      → stream audio to Jackie → start playback immediately
      → while generating next sentence's audio
    """
```

### 14. `session/manager.py` — Session Lifecycle

```python
class SessionManager:
    """
    Orchestrates the interaction lifecycle.
    
    Subscribes to: ENGAGEMENT_START, ENGAGEMENT_LOST, SPEAKER_VERIFIED,
                   SPEAKER_REJECTED, SESSION_END, FACE_LOST
    
    Behaviors:
    - On ENGAGEMENT_START: greet user proactively ("Hi! I'm Jackie...")
    - On SPEAKER_VERIFIED: route transcript to DialogueManager
    - On SPEAKER_REJECTED (N times): "Could you step a bit closer?"
    - On face lost: start grace timer (8s), if face returns → resume
    - On timeout/goodbye: farewell, cleanup, return to IDLE
    
    Session data: session_id, start_time, target_user_id, turn_count, 
                  all turns logged via DataLogger
    """
```

### 15. `utils/data_logger.py` — Interaction Logger

```python
class DataLogger:
    """
    Structured JSON logging for every interaction session.
    
    SessionLog:
      session_id, start_time, end_time, duration
      engagement_info (how session started, face area, gaze)
      cae_status (AEC/beamforming/NS on/off)
      doa_angles[] 
      turns[]:
        user_text, asr_confidence, asr_latency_ms
        asd_score, verification_result, verification_reason
        eou_confidence, silence_before_turn_ms
        robot_text, llm_latency_ms, llm_model_used
        tts_latency_ms, total_response_time_ms
      hri_success_checklist:
        engagement_detected, proactive_greeting, first_utterance_clean,
        multi_turn (≥3), no_phantoms, no_wrong_speaker, clean_farewell
      score: 0-7 (sum of checklist)
    
    Output: logs/<event-name>/<session-id>.json
    Also saves raw audio WAV per turn if configured.
    """
```

### 16. `main.py` — HRISystem Orchestrator

```python
class HRISystem:
    """
    Top-level asyncio application. Wires all modules together.
    
    Init order:
    1. Load config
    2. Init EventBus
    3. Init ConnectionManager (connect to Jackie)
    4. Init AudioPipeline + VideoPipeline
    5. Init all perception modules (ASR, FaceTracker, Gaze, ASD, Verifier, Engagement)
    6. Init EOUDetector
    7. Init DialogueManager
    8. Init TTS
    9. Init SessionManager
    10. Init DataLogger
    11. Subscribe all event handlers
    12. Start async loops
    
    Async loops:
    - _connection_loop(): manage Jackie connection
    - _audio_loop(): VAD → segments → ASR → EOU → verify → dialogue → TTS
    - _video_loop(): frames → face tracking → gaze → ASD → engagement
    
    Graceful shutdown: Ctrl+C → stop all modules → save logs → disconnect
    
    Crash recovery: auto-restart main loop on exception (max 10 retries, 1s cooldown)
    """
```

---

## 🚫 What NOT to Build (removed from v2)

- **Isaac Sim test infrastructure** — never used, dead code
- **SemanticVAD** (LLM-based hallucination filter) — replaced by confidence gate + EOU detector
- **ROS 2 integration** — not needed for conference demo
- **AWS Transcribe backend** — not using cloud ASR
- **`laser_asd1.py` duplicate** — one ASD module only
- **`network_sources.py` duplicate** — merged into connection/manager
- **Triple-layer hallucination filtering** — ONE clear filter: ASR confidence threshold
- **Behavior tree (py_trees)** — replaced by simpler SessionManager state machine. The BT added complexity without benefit for our use case.

---

## 🔑 Key Design Principles

1. **Event-driven architecture** — all modules communicate via EventBus. No tight coupling.
2. **Fail gracefully** — every module handles its own errors. One module crashing doesn't kill the system.
3. **Log everything** — every turn gets full latency breakdown. We need this data for the conference paper.
4. **Connection resilience** — auto-reconnect, fallback transports, heartbeat monitoring.
5. **VRAM-aware** — total budget is 12GB. Every model choice is deliberate.
6. **Conference-ready** — handles noise, bystanders, walk-aways, groups. Not a quiet-room demo.

---

## 📋 Config Values (tested + tuned)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| VAD threshold | 0.5 | Silero default, works well with CAE preprocessing |
| Min silence for EOU check | 300ms | Quick check, EOU decides if truly done |
| EOU confidence threshold | 0.7 | Below this, wait for more silence |
| Hard silence cutoff | 1500ms | Safety net if EOU is uncertain |
| ASR confidence gate | 0.40 | Short + low confidence = noise, not speech |
| ASD threshold | 0.5 | Confidence for accepting speaker |
| Session timeout | 30s | Conference: people move on quickly |
| Face lost grace | 8s | Handles brief look-aways |
| Reacquisition window | 20s | "I'll be right back" returns |
| Gaze engagement threshold | 2s sustained | Prevents greeting walk-bys |
| LLM max tokens | 150 | Keeps responses concise for spoken delivery |
| LLM temperature | 0.7 | Warm but not chaotic |
| Max history turns | 10 | Sliding window for context |
| Reconnect backoff | 1s→2s→4s→...→30s max | Exponential, caps at 30s |
| Heartbeat interval | 5s | Detect dead connections quickly |

---

## 🧪 Demo Scripts (`demo.py`)

Provide individual component test scripts:
- `python -m smait.demo audio` — test VAD with live mic, visual probability bar
- `python -m smait.demo asr` — test transcription live
- `python -m smait.demo tts` — test Kokoro TTS with text input
- `python -m smait.demo dialogue` — test LLM dialogue via text input
- `python -m smait.demo vision` — test face tracking + gaze + ASD with camera
- `python -m smait.demo connection` — test Jackie WebSocket connectivity
- `python -m smait.demo full` — run complete HRI system

---

## 📝 Requirements

```
# Core
numpy>=1.24.0
opencv-python>=4.8.0
python-dotenv>=1.0.0

# Audio
sounddevice>=0.4.6

# VAD
torch>=2.0.0
# silero-vad loaded via torch.hub

# ASR (Parakeet TDT via NeMo)
nemo_toolkit[asr]>=2.0.0

# TTS
kokoro>=0.8.0  # or whatever the pip package is

# Face tracking + Gaze
mediapipe>=0.10.0
# L2CS-Net (install from github)

# ASD
# TalkNet (install from github)

# LLM
openai>=1.0.0        # API fallback
requests>=2.31.0     # Ollama local API

# EOU detection
# livekit turn-detector (install from HuggingFace/github)

# WebSocket
websockets>=12.0

# Async
uvloop>=0.19.0; sys_platform != 'win32'

# Logging
psutil
```

---

## 🚀 Entry Point

```bash
# Production (with Jackie)
conda activate smait
python run_jackie.py --host 0.0.0.0 --port 8765

# With fallback URIs
python run_jackie.py --uris "ws://192.168.8.1:8765,ws://localhost:8765"

# Voice-only mode (no camera)
python run_jackie.py --voice-only

# Test connection only
python run_jackie.py --test
```

---

## ⚠️ Known Pitfalls (from v2 experience)

1. **NeMo 2.0 returns tuples** — Parakeet inference returns `(text, timestamps)` not just text. Handle the tuple.
2. **CUDA graph incompatibility on Blackwell** — Disable CUDA graphs for NeMo if on RTX 5070.
3. **Android WiFi routing** — Android may route traffic through mobile data even when on WiFi. The Jackie app must force WiFi route.
4. **Echo from TTS** — Gate the microphone (disable VAD) during TTS playback. Use TTS_START/TTS_END events.
5. **ASR hallucinations** — Parakeet can output "Thank you", "Bye", "Yeah" on silence/noise. Filter short low-confidence transcripts.
6. **Face ID reassignment** — After brief occlusion, the same person may get a new track ID. Use spatial + temporal matching for re-association.

---

## 🎯 Success Criteria

The system is ready for HFES when:
- [ ] Jackie connects reliably (travel router + USB fallback both work)
- [ ] Proactive greeting when someone approaches and looks at Jackie for >2s
- [ ] Correct speaker verification (rejects bystander speech)
- [ ] ASR latency < 200ms, total response time < 3s
- [ ] Natural-sounding TTS (Kokoro, not Android TTS)
- [ ] Clean session lifecycle (greet → converse → farewell)
- [ ] Data logger captures every interaction with full metrics
- [ ] Handles 10+ consecutive interactions without crashes
- [ ] Code is clean, documented, and explainable for conference Q&A
