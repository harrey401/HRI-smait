# SMAIT HRI v3.0 — Complete System Rebuild Prompt

> **One-shot rebuild of the entire SMAIT Human-Robot Interaction system.**
> **Date:** 2026-03-04 | **Target:** HFES Conference, mid-April 2026
> **Scope:** Python PC pipeline + Android Jackie app modifications

---

## 🎯 Project Overview

**SMAIT** (Social Multi-modal AI Interaction Technology) is a real-time Human-Robot Interaction system for **Jackie**, a mobile conference robot. It solves the **cocktail party problem** in HRI — isolating and understanding one target speaker in a noisy, crowded conference room (50+ people).

**What it does:**
1. Detects when someone approaches and wants to interact (engagement detection via gaze + proximity)
2. Captures raw multi-channel audio (4-mic array) AND hardware-processed audio simultaneously
3. Separates the target speaker's voice from crowd noise using Audio-Visual Target Speaker Extraction (Dolphin AV-TSE)
4. Transcribes the clean separated speech in real-time (Parakeet TDT ASR)
5. Generates a conversational response (LLM — local + API fallback)
6. Speaks the response back naturally (Kokoro TTS)
7. Manages the full interaction lifecycle with graceful session handling

**Environment:** Noisy conference floor, 50+ people in a small room, people moving table to table. Multiple simultaneous conversations. The robot sits at a table and must correctly isolate the person talking TO it.

**Academic context:** Masters project presented at HFES (Human Factors and Ergonomics Society) conference. Code must be clean, well-documented, and scientifically defensible.

---

## 🔧 Hardware

### Jackie Robot (Edge Device)
- **SoC:** Rockchip RK3588, 8-core ARM, 6 TOPS NPU, Android 12
- **Microphone:** 4-mic circular array with **iFLYTEK CAE SDK**
  - The CAE SDK provides: beamforming, AEC (acoustic echo cancellation), noise suppression, DOA (direction of arrival)
  - CAE takes raw mic input → outputs enhanced single-channel audio + DOA angle + wake word events
  - CAE code exists in the app (`CaeCoreHelper.java`) but is **currently NOT wired into the streaming pipeline**
  - The 4 physical mics can also be captured directly via Android AudioRecord with channel index mask
- **Camera:** Front-facing RGB camera (640×480 via Camera2 API)
- **Display:** 15.6" touchscreen
- **Speaker:** Built-in speaker for TTS playback
- **Connectivity:** WiFi (5GHz), USB-C

### Lab PC (AI Pipeline Server)
- **GPU:** NVIDIA RTX 5070 (12GB VRAM, Blackwell architecture, sm_120)
- **OS:** Ubuntu Linux, Python 3.10 (conda env: "smait")
- **CUDA:** Requires CUDA 12.8+ and PyTorch nightly for Blackwell support
- **All AI inference runs here** — Jackie is a sensor/actuator platform only

### Communication
- **Primary:** WebSocket over dedicated WiFi (GL.iNet travel router, isolated 5GHz network)
- **Fallback 1:** ADB USB port forwarding (`adb forward tcp:8765 tcp:8765`)
- **Fallback 2:** Phone hotspot as shared network
- **Protocol:** OkHttp WebSocket (Android) ↔ `websockets` library (Python)
  - Binary frames: type byte prefix (0x01=audio_cae, 0x02=video, 0x03=audio_raw_multichannel, 0x04=control)
  - Text frames: JSON control messages

---

## 📊 VRAM Budget (RTX 5070 — 12GB)

| Component | Model | Est. VRAM |
|-----------|-------|-----------|
| AV-TSE | Dolphin (speech separation) | ~2.0-3.0 GB |
| ASR | Parakeet TDT 0.6B v2 | ~2.0 GB |
| TTS | Kokoro-82M | ~1.0 GB |
| LLM | Phi-4 Mini 3.8B (Q4) | ~3.0 GB |
| Gaze | L2CS-Net | ~0.3 GB |
| Face | MediaPipe | ~0.2 GB (mostly CPU) |
| **Total** | | **~8.5-9.5 GB** |
| **Headroom** | | **~2.5-3.5 GB** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JACKIE (RK3588, Android 12)                    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 4-Mic Array                                                  │ │
│  │   ├─► Raw 4-channel capture (AudioRecord, channel index     │ │
│  │   │   mask 0x0F, 16kHz, PCM16) → stream as 0x03 frames     │ │
│  │   │                                                          │ │
│  │   └─► iFLYTEK CAE SDK (CaeCoreHelper)                      │ │
│  │       ├─► Enhanced single-channel audio → stream as 0x01    │ │
│  │       ├─► DOA angle (degrees) → JSON metadata               │ │
│  │       └─► AEC (echo cancellation from Jackie's speaker)     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │ Front Camera (Camera2)   │  │ Speaker                      │ │
│  │ 640×480 YUV → JPEG       │  │ ← TTS audio from PC          │ │
│  │ → stream as 0x02 frames  │  │ ← TTS_START/END gating       │ │
│  └──────────────────────────┘  └──────────────────────────────┘ │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ UI: Idle screen (pulse animation) ↔ Active screen (chat)    ││
│  │ State transitions driven by PC commands                      ││
│  │ Settings: IP/port config, volume, live parameter tuning      ││
│  │ Selfie feature: countdown → capture → save                   ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│         ↕ WebSocket (ws://travel-router-ip:8765)                 │
│         ↕ Fallback: ws://localhost:8765 (ADB USB)                │
└─────────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────────┐
│               LAB PC (RTX 5070, Ubuntu, Python 3.10)             │
│                                                                   │
│  ┌─ Connection Layer ────────────────────────────────────────┐  │
│  │ ConnectionManager: WebSocket server on 0.0.0.0:8765        │  │
│  │ • Accepts Jackie connection                                │  │
│  │ • Demuxes binary frames by type byte (0x01-0x04)           │  │
│  │ • Routes audio/video to respective pipelines               │  │
│  │ • Sends TTS audio + state commands back to Jackie          │  │
│  │ • Heartbeat ping/pong every 5s, auto-reconnect handling    │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                               │                                   │
│  ┌─ Audio Pipeline ──────────┼───────────────────────────────┐  │
│  │                           ▼                                │  │
│  │  0x01 (CAE audio) ──► Silero VAD                          │  │
│  │    Detects speech boundaries, emits SpeechSegments          │  │
│  │    VAD threshold: 0.5, min speech: 250ms                    │  │
│  │                           │                                │  │
│  │  0x03 (raw 4-ch) ──► Raw Audio Buffer                     │  │
│  │    Stored in time-aligned ring buffer (30s)                 │  │
│  │    When VAD triggers, extract matching time window           │  │
│  │                           │                                │  │
│  │                           ▼                                │  │
│  │  ┌─ Dolphin AV-TSE (GPU) ──────────────────────────────┐  │  │
│  │  │ Inputs:                                               │  │  │
│  │  │   • Raw 4-channel audio segment (for IPD/spatial)     │  │  │
│  │  │   • Lip ROI video frames (from face tracker)          │  │  │
│  │  │   • Target speaker face ID (from engagement)          │  │  │
│  │  │ Process:                                              │  │  │
│  │  │   • DP-LipCoder: lip video → discrete semantic tokens │  │  │
│  │  │   • Multi-channel encoder: 4-ch audio → spatial embed │  │  │
│  │  │   • Global-Local Attention separator                  │  │  │
│  │  │   • Visual tokens guide audio extraction              │  │  │
│  │  │ Output: clean isolated speech of target speaker       │  │  │
│  │  └──────────────────────────┬────────────────────────────┘  │  │
│  │                             ▼                                │  │
│  │  ┌─ DOA Verification ───────────────────────────────────┐  │  │
│  │  │ Cross-check: DOA angle from CAE vs face position      │  │  │
│  │  │ If DOA and face position diverge > 30° → flag/reject  │  │  │
│  │  │ Used as confidence boost, not hard gate                │  │  │
│  │  └──────────────────────────┬────────────────────────────┘  │  │
│  │                             ▼                                │  │
│  │  ┌─ ASR (Parakeet TDT 0.6B v2, GPU) ───────────────────┐  │  │
│  │  │ Input: clean separated audio from Dolphin             │  │  │
│  │  │ Output: TranscriptResult(text, confidence, timestamps)│  │  │
│  │  │ Confidence gate: < 0.40 on short utterances → reject  │  │  │
│  │  │ Hallucination filter: reject "Thank you" / "Bye" etc  │  │  │
│  │  └──────────────────────────┬────────────────────────────┘  │  │
│  │                             ▼                                │  │
│  │  ┌─ EOU Detector (LiveKit, CPU) ────────────────────────┐  │  │
│  │  │ Input: transcript text so far                         │  │  │
│  │  │ Output: P(end_of_turn)                                │  │  │
│  │  │ If silence > 300ms AND P(eou) > 0.7 → trigger LLM    │  │  │
│  │  │ Hard cutoff: silence > 1500ms → trigger anyway        │  │  │
│  │  └──────────────────────────┬────────────────────────────┘  │  │
│  └─────────────────────────────┼────────────────────────────────┘  │
│                                ▼                                   │
│  ┌─ Dialogue Layer ────────────────────────────────────────────┐  │
│  │ DialogueManager                                              │  │
│  │ • Primary: Phi-4 Mini 3.8B via Ollama (local, ~70 tok/s)    │  │
│  │ • Fallback: GPT-4o-mini API (if network available)           │  │
│  │ • Conversation memory: sliding window, 10 turns max          │  │
│  │ • System prompt: Jackie persona (warm, concise, owns being   │  │
│  │   a robot, 1-3 sentences max)                                │  │
│  │ • Goodbye detection → SESSION_END event                      │  │
│  │ • Streaming output for sentence-level TTS                    │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               ▼                                   │
│  ┌─ Output Layer ──────────────────────────────────────────────┐  │
│  │ Kokoro-82M TTS (GPU, <100ms TTFB, 96x real-time)            │  │
│  │ • Sentence-level streaming:                                   │  │
│  │   LLM streams tokens → detect sentence boundary (.!?)        │  │
│  │   → Kokoro generates audio for that sentence                  │  │
│  │   → Stream PCM bytes to Jackie via WebSocket                  │  │
│  │   → Jackie plays audio through speaker                        │  │
│  │ • Emit TTS_START event → tell Jackie to gate mic input        │  │
│  │ • Emit TTS_END event → tell Jackie to ungate mic              │  │
│  │ • This prevents robot hearing its own speech (echo loop)      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ Vision Pipeline (runs in parallel) ────────────────────────┐  │
│  │ 0x02 frames → decode JPEG                                    │  │
│  │   → MediaPipe Face Mesh (468 landmarks, multi-face)           │  │
│  │     → FaceTracker (persistent IDs across frames via IOU)      │  │
│  │     → Lip ROI extraction (mouth region crop for Dolphin)      │  │
│  │   → L2CS-Net gaze estimation (~0.3GB)                         │  │
│  │     → GazeResult(yaw, pitch, is_looking_at_robot)             │  │
│  │   → EngagementDetector                                        │  │
│  │     → Combines gaze + face area (distance proxy)              │  │
│  │     → Triggers engagement when both sustained > 2s            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ Session Manager ──────────────────────────────────────────┐  │
│  │ States: IDLE → APPROACHING → ENGAGED → CONVERSING          │  │
│  │         → DISENGAGING → IDLE                                │  │
│  │                                                              │  │
│  │ IDLE: no active interaction, vision loop scanning            │  │
│  │ APPROACHING: face detected, area growing, gaze toward robot  │  │
│  │ ENGAGED: sustained gaze > 2s → trigger proactive greeting    │  │
│  │ CONVERSING: active dialogue loop                             │  │
│  │ DISENGAGING: gaze away > 3s OR face area shrinking           │  │
│  │              OR silence > 30s → farewell → IDLE              │  │
│  │                                                              │  │
│  │ Rules:                                                       │  │
│  │ • Don't greet people walking past (rapidly changing face area)│  │
│  │ • Primary user = largest face + direct gaze + DOA alignment  │  │
│  │ • Face lost grace: 8s before ending session                  │  │
│  │ • Reacquisition window: 20s for "I'll be right back"         │  │
│  │ • On state change → send state JSON to Jackie app for UI     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─ Data Logger ──────────────────────────────────────────────┐  │
│  │ Structured JSON logs per interaction session                  │  │
│  │ Per turn: ASR confidence, ASD/Dolphin score, DOA angle,       │  │
│  │   separation SNR, EOU confidence, LLM latency, TTS latency,  │  │
│  │   total response time, model used                             │  │
│  │ HRI success checklist (auto-scored 0-7):                      │  │
│  │   engagement_detected, proactive_greeting,                    │  │
│  │   first_utterance_clean, multi_turn (≥3),                     │  │
│  │   no_phantoms, no_wrong_speaker, clean_farewell               │  │
│  │ Output: logs/<event-name>/<session-id>.json                   │  │
│  │ Optional: save raw audio WAV + separated audio per turn       │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Python Package Structure (PC Side)

```
smait/
├── core/
│   ├── __init__.py
│   ├── config.py              — Dataclass config hierarchy, env/file loading, singleton
│   └── events.py              — Async EventBus pub/sub (ALL inter-module communication)
├── connection/
│   ├── __init__.py
│   ├── manager.py             — WebSocket server, connection handling, frame demuxing
│   └── protocol.py            — Frame types, message schemas, serialization
├── sensors/
│   ├── __init__.py
│   ├── audio_pipeline.py      — Silero VAD + ring buffer → SpeechSegment
│   └── video_pipeline.py      — JPEG decode, frame buffer, frame distribution
├── perception/
│   ├── __init__.py
│   ├── face_tracker.py        — MediaPipe Face Mesh, multi-face, persistent track IDs
│   ├── lip_extractor.py       — Extract mouth ROI crops from face landmarks for Dolphin
│   ├── gaze.py                — L2CS-Net gaze estimation
│   ├── engagement.py          — Engagement detection (gaze + face area + debounce)
│   ├── dolphin_separator.py   — Dolphin AV-TSE: lip video + multi-channel audio → clean speech
│   ├── asr.py                 — Parakeet TDT 0.6B v2 wrapper (NeMo, GPU)
│   ├── transcriber.py         — ASR orchestration: separated audio → transcript + filters
│   └── eou_detector.py        — LiveKit End-of-Utterance model (CPU, text-based)
├── dialogue/
│   ├── __init__.py
│   └── manager.py             — LLM dialogue (Phi-4 local + API fallback), conversation memory
├── output/
│   ├── __init__.py
│   └── tts.py                 — Kokoro-82M TTS, sentence-level streaming, mic gating
├── session/
│   ├── __init__.py
│   └── manager.py             — Session lifecycle state machine, proactive behaviors
├── utils/
│   ├── __init__.py
│   ├── data_logger.py         — Structured JSON interaction logger + HRI checklist
│   └── metrics.py             — Latency/performance metrics capture
├── __init__.py
├── main.py                    — HRISystem: wires everything, runs async loops
└── demo.py                    — Individual component test scripts
```

---

## 🤖 Android App Modifications (Jackie Side)

The existing app is at `smait-jackie-app/`. It's a Kotlin Android app using OkHttp WebSocket, Camera2 API, and Android TTS. The following modifications are needed:

### Current App State (what exists)
- `MainActivity.kt` — full UI with idle/active screens, chat, selfie feature, settings
- `CaeCoreHelper.java` — iFLYTEK CAE SDK wrapper (exists but NOT wired into streaming)
- Audio: `AudioRecord` with `VOICE_COMMUNICATION` source, **MONO**, 16kHz → streams as 0x01 binary frames
- Video: Camera2 API, 640×480 YUV → JPEG (quality 75) → streams as 0x02 binary frames
- WebSocket: OkHttp client with auto-reconnect (exponential backoff 1s→30s)
- TTS: Android `TextToSpeech` (needs to be replaced with receiving audio from PC)
- UI states: IDLE ↔ ENGAGED, driven by server JSON messages

### Required Modifications

#### 1. Activate iFLYTEK CAE SDK
The `CaeCoreHelper.java` and `OnCaeOperatorlistener.java` already exist. Wire them in:

```kotlin
// In MainActivity.kt, add:
private lateinit var caeHelper: CaeCoreHelper
private var lastDoaAngle: Int = -1

// In onCreate or after permissions:
CaeCoreHelper.portingFile(this)  // copy model files to /sdcard/cae/
caeHelper = CaeCoreHelper(object : OnCaeOperatorlistener {
    override fun onWakeup(angle: Int, beam: Int) {
        lastDoaAngle = angle
        // Send DOA to PC
        webSocket?.send(JSONObject().apply {
            put("type", "doa")
            put("angle", angle)
            put("beam", beam)
        }.toString())
    }
    
    override fun onAudio(audioData: ByteArray, dataLen: Int) {
        // CAE-processed audio → stream as 0x01
        if (isStreaming.get() && !isSpeaking.get()) {
            val frame = ByteArray(1 + dataLen)
            frame[0] = 0x01  // CAE audio type
            System.arraycopy(audioData, 0, frame, 1, dataLen)
            webSocket?.send(frame.toByteString(0, frame.size))
        }
    }
}, false)  // false = 4-mic mode (not 2-mic)
```

#### 2. Add Raw 4-Channel Audio Capture
Capture raw multi-channel audio simultaneously for Dolphin's spatial features:

```kotlin
// New constants
private const val RAW_CHANNEL_CONFIG = 0x0F  // Channel index mask for 4 channels
private const val RAW_AUDIO_TYPE: Byte = 0x03

// New AudioRecord for raw capture
private var rawAudioRecord: AudioRecord? = null
private var rawAudioThread: Thread? = null

private fun startRawAudioCapture() {
    val bufferSize = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,  // Will be overridden by channel index mask
        AUDIO_FORMAT
    ) * 4  // 4 channels
    
    rawAudioRecord = AudioRecord.Builder()
        .setAudioSource(MediaRecorder.AudioSource.UNPROCESSED)  // Raw, no AGC/NS
        .setAudioFormat(AudioFormat.Builder()
            .setSampleRate(SAMPLE_RATE)
            .setEncoding(AUDIO_FORMAT)
            .setChannelIndexMask(0x0F)  // 4 channels via index mask
            .build())
        .setBufferSizeInBytes(bufferSize)
        .build()
    
    rawAudioRecord?.startRecording()
    
    rawAudioThread = Thread {
        val buffer = ByteArray(bufferSize)
        while (isStreaming.get()) {
            val read = rawAudioRecord?.read(buffer, 0, buffer.size) ?: 0
            if (read > 0 && !isSpeaking.get()) {
                val frame = ByteArray(1 + read)
                frame[0] = RAW_AUDIO_TYPE  // 0x03
                System.arraycopy(buffer, 0, frame, 1, read)
                webSocket?.send(frame.toByteString(0, frame.size))
            }
        }
    }.also { it.start() }
}
```

**Important:** The `UNPROCESSED` audio source may not be supported on all RK3588 OEM builds. If it fails, fall back to `VOICE_RECOGNITION` (disables AGC/NS but may retain linear AEC). Test on the actual device.

**Also important:** The raw 4-channel capture and CAE processing should use the same physical mics. The CAE SDK reads from the mic array internally via its native layer — it doesn't conflict with a separate `AudioRecord` instance because the CAE SDK uses its own audio HAL path. However, if there IS a conflict on this specific RK3588 board, fall back to feeding raw audio TO the CAE via `caeHelper.writeAudio(buffer)` and only streaming the CAE output (0x01) + using DOA angles. Dolphin can work with single-channel audio + visual features — the 4-channel IPD is a bonus.

#### 3. Feed Raw Audio to CAE Engine
The CAE SDK processes audio fed to it via `writeAudio()`. Route the raw mono capture through it:

```kotlin
// In the existing audio capture thread, add:
// After reading from audioRecord, also feed to CAE
if (read > 0) {
    caeHelper.writeAudio(buffer.copyOf(read))
}
```

#### 4. Replace Android TTS with PC Audio Playback
Currently the app uses Android's `TextToSpeech` engine (robotic quality). Replace with receiving Kokoro TTS audio from the PC:

```kotlin
// New: AudioTrack for playing TTS audio from PC
private var audioTrack: AudioTrack? = null

private fun initAudioPlayback() {
    val bufferSize = AudioTrack.getMinBufferSize(
        24000,  // Kokoro outputs 24kHz
        AudioFormat.CHANNEL_OUT_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )
    audioTrack = AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build())
        .setAudioFormat(AudioFormat.Builder()
            .setSampleRate(24000)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build())
        .setBufferSizeInBytes(bufferSize)
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()
}

// Handle incoming TTS audio binary frames (type 0x05)
// In WebSocket onMessage for binary:
override fun onMessage(ws: WebSocket, bytes: ByteString) {
    val data = bytes.toByteArray()
    if (data.isNotEmpty()) {
        when (data[0]) {
            0x05.toByte() -> {  // TTS audio from PC
                val audioData = data.copyOfRange(1, data.size)
                audioTrack?.write(audioData, 0, audioData.size)
                if (audioTrack?.playState != AudioTrack.PLAYSTATE_PLAYING) {
                    audioTrack?.play()
                }
            }
        }
    }
}
```

Keep the existing Android TTS as a fallback in case the PC TTS pipeline fails.

#### 5. Binary Frame Protocol Update

```
Inbound (Jackie → PC):
  0x01 = CAE-processed audio (single channel, 16kHz, PCM16)
  0x02 = Video frame (JPEG bytes)
  0x03 = Raw 4-channel audio (interleaved, 16kHz, PCM16, 4ch)
  
  JSON text frames:
    {"type": "doa", "angle": <int>, "beam": <int>}
    {"type": "tts_state", "speaking": <bool>}
    {"type": "config", "vad_threshold": ..., "asd_min_score": ..., "session_timeout": ...}
    {"type": "cae_status", "aec": <bool>, "beamforming": <bool>, "noise_suppression": <bool>}

Outbound (PC → Jackie):
  0x05 = TTS audio (24kHz, mono, PCM16)
  
  JSON text frames:
    {"type": "state", "state": "idle"|"engaged", "robot_status": "listening"|"thinking"|"speaking"}
    {"type": "transcript", "text": "...", "speaker": "user"|"robot"}
    {"type": "tts", "text": "..."} (fallback: use Android TTS if PC TTS fails)
    {"type": "tts_control", "action": "start"|"end"} (mic gating signal)
    {"type": "response", "text": "..."} (adds to chat + triggers TTS)
```

---

## 🧩 Detailed Module Specifications (Python PC Side)

### 1. `core/config.py`

```python
@dataclass
class ConnectionConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    heartbeat_interval_s: float = 5.0
    reconnect_max_wait_s: float = 30.0

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels_raw: int = 4            # Raw multi-channel from Jackie
    channels_cae: int = 1            # CAE-processed single channel
    chunk_duration_ms: int = 30      # VAD chunk size
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    raw_buffer_seconds: float = 30.0  # Ring buffer for raw 4-ch audio

@dataclass
class SeparationConfig:
    model: str = "dolphin"           # AV-TSE model
    use_multichannel: bool = True    # Use raw 4-ch for IPD features
    fallback_to_cae: bool = True     # If no raw 4-ch, use CAE audio directly

@dataclass
class ASRConfig:
    model: str = "nvidia/parakeet-tdt-0.6b-v2"
    confidence_threshold: float = 0.40
    hallucination_filter: bool = True  # Reject known phantom phrases

@dataclass
class EOUConfig:
    min_silence_ms: int = 300        # Check EOU after this much silence
    confidence_threshold: float = 0.7
    hard_cutoff_ms: int = 1500       # Force trigger after this silence

@dataclass
class VisionConfig:
    max_faces: int = 5
    min_face_confidence: float = 0.6
    lip_roi_size: tuple = (96, 96)   # Mouth crop size for Dolphin

@dataclass
class GazeConfig:
    yaw_threshold: float = 30.0      # degrees
    pitch_threshold: float = 20.0

@dataclass
class EngagementConfig:
    min_gaze_duration_s: float = 2.0  # Sustained gaze before greeting
    disengage_gaze_timeout_s: float = 3.0
    face_area_threshold: int = 3000   # Min face bbox area (pixels²)

@dataclass
class DialogueConfig:
    local_model: str = "phi-4-mini"   # Ollama model name
    api_model: str = "gpt-4o-mini"
    api_provider: str = "openai"
    max_tokens: int = 150
    temperature: float = 0.7
    max_history_turns: int = 10
    try_local_first: bool = True
    system_prompt: str = (
        "You are Jackie, a friendly AI-powered conference robot at SJSU. "
        "Keep responses to 1-3 spoken sentences. Be warm, natural, slightly playful. "
        "You ARE a robot — own it with personality. "
        "You can see faces, detect who's speaking, and hold real conversations in real-time. "
        "If speech seems garbled: 'Sorry, it's noisy — could you say that again?' "
        "Plain spoken sentences only — no lists, markdown, or formatting."
    )

@dataclass
class TTSConfig:
    model: str = "kokoro-82m"
    sample_rate: int = 24000         # Kokoro output sample rate
    stream_by_sentence: bool = True

@dataclass
class SessionConfig:
    timeout_s: float = 30.0
    face_lost_grace_s: float = 8.0
    reacquisition_window_s: float = 20.0

@dataclass
class LogConfig:
    output_dir: str = "logs"
    save_audio: bool = True          # Save raw + separated audio per turn
    save_video_snapshots: bool = False

@dataclass
class Config:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    eou: EOUConfig = field(default_factory=EOUConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    debug: bool = False
    show_video: bool = True
```

### 2. `core/events.py` — Event Bus

ALL inter-module communication goes through EventBus. No module directly calls another.

```python
class EventType(Enum):
    # Audio
    SPEECH_DETECTED = auto()         # VAD triggered
    SPEECH_SEGMENT = auto()          # Complete speech segment ready
    
    # Separation
    SPEECH_SEPARATED = auto()        # Dolphin output: clean target audio
    
    # ASR
    TRANSCRIPT_READY = auto()        # TranscriptResult with text + confidence
    TRANSCRIPT_REJECTED = auto()     # Filtered out (low confidence / hallucination)
    
    # Turn-taking
    END_OF_TURN = auto()             # EOU says user is done speaking
    
    # Vision
    FACE_DETECTED = auto()           # New face entered frame
    FACE_LOST = auto()               # Tracked face disappeared
    FACE_UPDATED = auto()            # Landmarks/pose updated
    GAZE_UPDATE = auto()             # Per-face gaze result
    LIP_ROI_READY = auto()           # Mouth crop ready for Dolphin
    
    # Engagement
    ENGAGEMENT_START = auto()
    ENGAGEMENT_LOST = auto()
    
    # Dialogue
    DIALOGUE_RESPONSE = auto()       # LLM response text
    DIALOGUE_STREAM = auto()         # Partial LLM response (for streaming TTS)
    
    # TTS
    TTS_START = auto()               # Robot speaking → gate mic
    TTS_END = auto()                 # Robot done → ungate mic
    TTS_AUDIO_CHUNK = auto()         # Audio bytes to send to Jackie
    
    # Session
    SESSION_START = auto()
    SESSION_END = auto()
    
    # Hardware / Connection
    DOA_UPDATE = auto()              # angle, beam from CAE
    CAE_STATUS = auto()              # Hardware filter states
    CONNECTION_OPEN = auto()
    CONNECTION_CLOSED = auto()
    
    # System
    ERROR = auto()

class EventBus:
    """Async pub/sub. Supports both coroutine and sync handlers."""
    def subscribe(self, event_type: EventType, handler: Callable): ...
    def emit(self, event_type: EventType, data: Any = None): ...
    async def emit_async(self, event_type: EventType, data: Any = None): ...
```

### 3. `connection/manager.py` — WebSocket Server

```python
class ConnectionManager:
    """
    WebSocket server that accepts Jackie's connection.
    
    - Listens on 0.0.0.0:8765
    - Single client connection (one robot)
    - Demuxes incoming binary frames by type byte:
        0x01 → CAE audio → AudioPipeline
        0x02 → Video JPEG → VideoPipeline  
        0x03 → Raw 4-ch audio → RawAudioBuffer
    - Handles incoming JSON messages (DOA, TTS state, config, CAE status)
    - Provides send methods for outbound data:
        send_tts_audio(pcm_bytes) → 0x05 binary frame
        send_state(state_dict) → JSON text frame
        send_transcript(text, speaker) → JSON text frame
    - Heartbeat: ping every 5s, timeout after 15s → CONNECTION_CLOSED
    - Emits CONNECTION_OPEN / CONNECTION_CLOSED events
    """
```

### 4. `sensors/audio_pipeline.py` — VAD

```python
class AudioPipeline:
    """
    Receives CAE-processed audio (single channel, 16kHz).
    Runs Silero VAD on 30ms chunks.
    When speech detected: accumulates in buffer.
    When silence detected (> min_silence): emits SPEECH_SEGMENT.
    
    Also maintains a synchronized pointer into the RawAudioBuffer
    so that when a speech segment is identified, the corresponding
    raw 4-channel audio window can be extracted for Dolphin.
    
    Mic gating: when TTS_START event received, disable VAD output
    (prevent robot hearing itself). Re-enable on TTS_END.
    
    SpeechSegment = {
        cae_audio: np.ndarray,     # CAE-processed (for fallback ASR)
        raw_audio: np.ndarray,     # Raw 4-channel (for Dolphin)
        start_time: float,
        end_time: float,
        duration: float
    }
    """
```

### 5. `perception/dolphin_separator.py` — AV-TSE

```python
class DolphinSeparator:
    """
    Dolphin Audio-Visual Target Speaker Extraction.
    GitHub: JusperLee/Dolphin
    
    This is the core innovation — it simultaneously:
    1. Identifies who is speaking (via lip-audio correlation)
    2. Separates their voice from the mix (source separation)
    
    Inputs:
    - audio: raw 4-channel PCM segment (or single-channel CAE fallback)
    - lip_frames: sequence of mouth ROI crops for the target face
      (extracted by lip_extractor from face tracker landmarks)
    - target_face_id: which face to extract speech for
    
    Process:
    1. DP-LipCoder: lip video → discrete semantic tokens via vector quantization
       (strips lighting, head pose — keeps only articulatory info)
    2. Multi-channel audio encoder: computes Inter-channel Phase Differences (IPD)
       across the 4 mic channels → spatial embedding
       (tells the model WHERE each sound source is in 3D space)
    3. Global-Local Attention (GLA) separator:
       - Global: coarse self-attention at low resolution (long-range dependencies)
       - Local: heat diffusion smoothing (noise suppression)
    4. Visual tokens guide the audio decoder — extract only acoustic features
       that temporally align with the target speaker's lip movement
    
    Output:
    - separated_audio: np.ndarray (clean speech of target speaker, 16kHz mono)
    - separation_confidence: float (SNR improvement estimate)
    
    Fallback: If no raw 4-channel audio available, Dolphin works with
    single-channel audio + visual features (reduced spatial discrimination
    but still effective for temporal/articulatory separation).
    
    Performance: 6x real-time on GPU, ~2-3GB VRAM.
    50% fewer parameters than prior SOTA (efficient for RTX 5070).
    """
```

### 6. `perception/lip_extractor.py` — Mouth ROI

```python
class LipExtractor:
    """
    Extracts mouth region-of-interest (ROI) crops from face tracker output.
    
    Input: FaceTrack with 468 MediaPipe landmarks
    Output: lip_roi (96×96 RGB crop centered on mouth region)
    
    Uses MediaPipe landmarks 61-68 (outer lips) and 78-95 (inner lips)
    to compute a bounding box around the mouth, then crops and resizes.
    
    Maintains a temporal buffer of lip ROI frames per tracked face,
    synchronized with audio timestamps, for feeding to Dolphin.
    
    Emits LIP_ROI_READY events with the buffer when speech segment arrives.
    """
```

### 7. `perception/asr.py` — Parakeet TDT

```python
class ParakeetASR:
    """
    NVIDIA NeMo Parakeet TDT 0.6B v2.
    
    - Load model once at init (GPU, ~2GB VRAM)
    - transcribe(audio: np.ndarray) → TranscriptResult
    - Input: clean separated audio from Dolphin (16kHz mono float32)
    - Output: TranscriptResult(text, confidence, word_timestamps, latency_ms)
    
    NeMo 2.0 quirks:
    - Returns tuple (text, timestamps), not just text
    - Disable CUDA graphs on Blackwell (sm_120) to avoid kernel errors
    - Model downloaded from HuggingFace on first run (~600MB)
    
    Confidence normalization: NeMo logprobs → 0-1 scale.
    """
```

### 8. `perception/transcriber.py` — ASR Pipeline

```python
class Transcriber:
    """
    Orchestrates the ASR pipeline:
    
    1. Receive SPEECH_SEPARATED event (clean audio from Dolphin)
    2. Run Parakeet ASR → TranscriptResult
    3. Apply confidence filter:
       - Short utterance (<8 words) + confidence < 0.40 → REJECT
       - Known hallucination phrases ("Thank you.", "Bye.", "Yeah.") 
         at low confidence → REJECT
    4. If passes: emit TRANSCRIPT_READY
    5. If rejected: emit TRANSCRIPT_REJECTED with reason
    
    TranscriptResult = {
        text: str,
        confidence: float,
        word_timestamps: list,
        latency_ms: float,
        start_time: float,
        end_time: float
    }
    """
```

### 9. `perception/eou_detector.py` — Turn-Taking

```python
class EOUDetector:
    """
    LiveKit open-weight End-of-Utterance model.
    GitHub: livekit/turn-detector (HuggingFace)
    
    Runs on CPU (no VRAM cost).
    
    Usage flow:
    1. AudioPipeline detects silence > 300ms after speech
    2. Current transcript fed to EOUDetector
    3. Model returns P(end_of_turn) ∈ [0, 1]
    4. If P > 0.7 → emit END_OF_TURN → triggers LLM response
    5. If P < 0.7 → wait for more speech or silence
    6. Safety: if silence > 1500ms regardless of P → force END_OF_TURN
    
    This reduces effective response latency from 1000ms to ~300-500ms
    for clear end-of-turn cases ("What time does the keynote start?")
    while preventing premature cutoffs during thinking pauses
    ("I was wondering if... [pause] ...you could help me find...").
    """
```

### 10. `perception/face_tracker.py`

```python
class FaceTracker:
    """
    MediaPipe Face Mesh: 468 landmarks per face, 30+ FPS.
    
    - Persistent track IDs via IOU/centroid matching across frames
    - Re-association after brief occlusion (spatial + temporal matching)
    - FaceTrack dataclass:
        track_id, bbox, landmarks, head_yaw, head_pitch,
        last_seen, confidence, is_target, face_area
    - Emits FACE_DETECTED, FACE_LOST, FACE_UPDATED events
    - Max 5 simultaneous tracked faces
    """
```

### 11. `perception/gaze.py`

```python
class GazeEstimator:
    """
    L2CS-Net: lightweight gaze estimation (~0.3GB VRAM).
    
    Input: face crop from FaceTracker bbox
    Output: GazeResult(yaw_deg, pitch_deg, is_looking_at_robot)
    
    is_looking_at_robot = |yaw| < 30° AND |pitch| < 20°
    Emits GAZE_UPDATE per tracked face per frame.
    """
```

### 12. `perception/engagement.py`

```python
class EngagementDetector:
    """
    Determines if someone wants to interact.
    
    Two signals:
    1. Distance proxy: face bounding box area (larger = closer)
    2. Gaze: L2CS-Net is_looking_at_robot
    
    Engagement triggered when BOTH sustained > 2 seconds (debounced).
    
    Rules:
    - Rapidly changing face area = walking past → don't engage
    - Multiple faces: primary = largest face + direct gaze + DOA alignment
    - Group of people: wait for individual to step forward
    
    State machine:
    IDLE → APPROACHING (face detected + area growing + gaze toward)
    APPROACHING → ENGAGED (sustained > 2s → emit ENGAGEMENT_START)
    ENGAGED → LOST (gaze away > 3s OR area shrinking → emit ENGAGEMENT_LOST)
    """
```

### 13. `dialogue/manager.py`

```python
class DialogueManager:
    """
    Hybrid LLM: Phi-4 Mini (local, Ollama) + GPT-4o-mini (API fallback).
    
    - try_local_first: attempts Ollama first, falls back to API on timeout/error
    - Conversation memory: sliding window of last 10 turns
    - System prompt: Jackie persona
    - Goodbye detection: analyze response for farewell intent → SESSION_END
    - ask_async(text) → DialogueResponse(text, latency_ms, model_used, tokens_used)
    - ask_streaming(text) → AsyncGenerator[str]: yields partial responses
      for sentence-level TTS (generate TTS for sentence 1 while LLM produces sentence 2)
    """
```

### 14. `output/tts.py`

```python
class TTSEngine:
    """
    Kokoro-82M: <100ms TTFB, ~1GB VRAM, 96x real-time.
    
    Sentence-level streaming pipeline:
    1. Subscribe to DIALOGUE_STREAM events (partial LLM output)
    2. Buffer tokens until sentence boundary detected (.!?)
    3. Generate audio for that sentence with Kokoro
    4. Emit TTS_AUDIO_CHUNK → ConnectionManager sends to Jackie
    5. Jackie plays audio through AudioTrack
    
    Mic gating:
    - Before first chunk: emit TTS_START → Jackie gates mic
    - After last chunk: emit TTS_END → Jackie ungates mic
    
    Fallback: if Kokoro fails, send text via JSON {"type": "tts", "text": "..."}
    and let Jackie use Android TTS.
    """
```

### 15. `session/manager.py`

```python
class SessionManager:
    """
    Orchestrates interaction lifecycle.
    
    Subscribes to: ENGAGEMENT_START, ENGAGEMENT_LOST, END_OF_TURN,
                   TRANSCRIPT_READY, DIALOGUE_RESPONSE, TTS_END,
                   FACE_LOST, CONNECTION_CLOSED
    
    State machine: IDLE → APPROACHING → ENGAGED → CONVERSING → DISENGAGING → IDLE
    
    Behaviors:
    - ENGAGEMENT_START → set target_user_id, greet proactively, switch to CONVERSING
    - END_OF_TURN → route transcript to DialogueManager
    - DIALOGUE_RESPONSE → route to TTSEngine
    - FACE_LOST → start grace timer (8s), if face returns → resume
    - Silence > 30s → farewell, cleanup, IDLE
    - Goodbye detected in LLM response → farewell, cleanup, IDLE
    - On each state change → send state JSON to Jackie for UI update
    - Log everything via DataLogger
    """
```

### 16. `utils/data_logger.py`

```python
class DataLogger:
    """
    Structured JSON per session.
    
    SessionLog:
      session_id, start_time, end_time, duration
      engagement_info: {face_area, gaze_duration, doa_angle}
      cae_status: {aec, beamforming, noise_suppression}
      doa_angles: [int]
      turns: [
        {
          user_text, asr_confidence, asr_latency_ms,
          separation_snr, dolphin_confidence,
          doa_angle, doa_face_alignment_deg,
          eou_confidence, silence_before_turn_ms,
          robot_text, llm_latency_ms, llm_model_used,
          tts_latency_ms, total_response_time_ms,
          verification_result, verification_reason
        }
      ]
      hri_checklist: {
        engagement_detected, proactive_greeting,
        first_utterance_clean, multi_turn_3plus,
        no_phantoms, no_wrong_speaker, clean_farewell
      }
      score: 0-7
    
    Output: logs/<event>/<session-id>.json
    Optional: save WAV files (raw, separated, response audio)
    """
```

### 17. `main.py` — HRISystem

```python
class HRISystem:
    """
    Top-level asyncio application.
    
    Init order:
    1. Load Config (from env/file)
    2. Create EventBus
    3. Start ConnectionManager (WebSocket server)
    4. Init AudioPipeline + VideoPipeline + RawAudioBuffer
    5. Init FaceTracker + LipExtractor + GazeEstimator + EngagementDetector
    6. Init DolphinSeparator (GPU)
    7. Init ParakeetASR (GPU)
    8. Init Transcriber
    9. Init EOUDetector (CPU)
    10. Init DialogueManager (GPU for local / API for cloud)
    11. Init TTSEngine (GPU)
    12. Init SessionManager
    13. Init DataLogger
    14. Wire all event subscriptions
    15. Start async loops
    
    Async loops:
    - _connection_loop(): manage Jackie WebSocket
    - _audio_loop(): CAE audio → VAD → speech segments
    - _video_loop(): JPEG frames → face tracking → gaze → engagement → lip ROI
    - _separation_loop(): speech segment + lip ROIs → Dolphin → clean audio
    - _asr_loop(): clean audio → Parakeet → transcript → EOU → dialogue → TTS
    
    Crash recovery: auto-restart each loop on exception (max 10 retries, 1s cooldown).
    Graceful shutdown: Ctrl+C → stop all → save logs → disconnect.
    """
```

### 18. `demo.py` — Component Tests

```python
"""
Individual component demo scripts.
Usage: python -m smait.demo <component>

Components:
  connection  — test Jackie WebSocket connectivity + frame reception
  audio       — test VAD with live audio, show probability bar
  video       — test face tracking + gaze + lip ROI with camera display
  separation  — test Dolphin with live audio + video
  asr         — test Parakeet transcription on clean audio
  tts         — test Kokoro TTS with text input
  dialogue    — test LLM dialogue via text input
  eou         — test end-of-utterance detection with typed sentences
  full        — run complete HRI system
"""
```

---

## 🚫 What NOT to Build (removed from v2)

- **Isaac Sim test infrastructure** — never used
- **SemanticVAD** (LLM-based hallucination filter) — Dolphin + confidence gate replaces this
- **ROS 2 integration** — not needed
- **AWS Transcribe backend** — not using cloud ASR
- **`laser_asd.py` / `laser_asd1.py`** — Dolphin replaces LASER ASD entirely
- **`network_sources.py` duplicate** — merged into connection/manager
- **Behavior tree (py_trees)** — SessionManager state machine is simpler and sufficient
- **Faster-whisper fallback** — Parakeet is the only ASR backend needed
- **`sources.py` camera abstraction** — video comes from WebSocket only, no local camera

---

## ⚠️ Known Pitfalls & Lessons from v2

### Critical Issues from Previous Versions (all 8 documented bugs)

**Issue #6 — Parakeet "yeah" hallucination (CRITICAL, caused real speech loss in v2):**
- Parakeet TDT sometimes REPLACES an entire real utterance with "yeah", "okay", or other filler words
- Root cause: audio segment too short (VAD cut too early) → Parakeet has insufficient context → hallucinates
- **Mitigations in v3:**
  1. LiveKit EOU replaces dumb silence timeout → segments are longer and more complete
  2. Dolphin separation cleans audio BEFORE ASR → higher quality input
  3. **Minimum segment length check: reject segments < 500ms** (too short for real speech)
  4. Confidence gate: short utterance (<8 words) + confidence < 0.40 → reject
  5. Hallucination phrase blocklist: if transcript is ONLY "yeah"/"okay"/"thank you"/"bye"/"thanks for watching" AND confidence < 0.60 → reject
  6. **Save raw audio WAV per turn** (even rejected ones) for debugging

**Issue #2 & #4 — Premature cutoffs / short word drops:**
- In v2, silence_duration_ms=500 cut off slow speakers mid-thought and isolated short words
- v3 fix: EOU model handles this properly — checks semantic completeness, not just silence

**Issue #3 — "User not visible" during active speech:**
- In v2, threshold mismatch (0.3 vs 0.2) in the verifier caused false rejections
- v3 fix: Dolphin replaces the entire ASD verification pipeline — no manual thresholds to mismatch

**Issue #5 — TTS sounds robotic (Android stock TTS):**
- v3 fix: Kokoro-82M audio streamed from PC to Jackie via binary WebSocket frames

**Issue #1 — Background noise / CAE not validated:**
- In v2, `cae_status` was received but never checked — system didn't know if beamforming was actually ON
- v3 fix: On connection, request CAE status. If beamforming/AEC/NS are OFF, log a WARNING and optionally alert operator. Don't silently proceed with degraded audio.

### Technical Pitfalls

1. **NeMo 2.0 tuple returns** — Parakeet inference returns `(text, timestamps)` not just text. Handle the tuple.
2. **Blackwell CUDA graphs** — Disable CUDA graphs for NeMo on sm_120 (RTX 5070). Set `NEMO_DISABLE_CUDA_GRAPHS=1` env var.
3. **Android UNPROCESSED audio source** — May not be supported on RK3588 OEM build. Fallback: `VOICE_RECOGNITION` (disables AGC/NS but may retain linear AEC, which is tolerable).
4. **4-channel capture conflict with CAE** — If both can't access mics simultaneously on this specific RK3588 board, feed raw audio to CAE via `writeAudio()` and only stream CAE output (0x01) + DOA angles. Dolphin can work with single-channel + visual features (reduced spatial discrimination but still effective).
5. **Echo from TTS** — Gate microphone during TTS playback using TTS_START/TTS_END events. CAE's AEC also helps but is not sufficient alone — the robot's speaker is very close to its mics.
6. **Face ID reassignment** — After brief occlusion, same person may get new track ID. Use spatial + temporal matching for re-association. If target face is lost and a new face appears at similar position within 2s, re-associate.
7. **Android WiFi routing** — Android may route traffic through mobile data even when on WiFi if the WiFi has no internet. Use travel router where Jackie gets no internet route → forces all traffic through WiFi. Alternatively, in Android developer settings, disable "mobile data always active".
8. **Kokoro word drops** — Known issue on long sentences. Sentence-level streaming mitigates this (each Kokoro input is 1 sentence = short).
9. **Dolphin model availability** — Check JusperLee/Dolphin GitHub for pretrained weights. If unavailable, fall back to SpeechBrain SepFormer (similar VRAM, less visual integration) + TalkNet ASD (for speaker identification separately).
10. **WebSocket binary frame ordering** — Audio and video frames may arrive out of order under load. Use timestamps (not arrival order) for synchronization. Each frame should carry a timestamp from Jackie's system clock.
11. **CAE SDK initialization timing** — `CaeCoreHelper.portingFile()` copies model files to `/sdcard/cae/`. This must happen BEFORE `CAENew()` is called. On first install, files may not exist yet. Handle gracefully.
12. **Concurrent AudioRecord instances** — Having two AudioRecord instances (one for raw 4-ch, one for CAE feed) may conflict on some Android devices. If the second AudioRecord fails to initialize, fall back to single-stream mode (CAE only + DOA).

---

## 📝 Dependencies

```
# Core
numpy>=1.24.0
opencv-python>=4.8.0
python-dotenv>=1.0.0
websockets>=12.0

# Audio
sounddevice>=0.4.6

# VAD
torch>=2.0.0  # Also used by Dolphin, NeMo, Kokoro
# silero-vad via torch.hub

# Speech Separation (Dolphin AV-TSE)
# Install from: github.com/JusperLee/Dolphin
# Dependencies: torch, torchaudio, einops

# ASR
nemo_toolkit[asr]>=2.0.0

# TTS
# kokoro (install from HuggingFace: hexgrad/Kokoro-82M)

# Face tracking + Gaze
mediapipe>=0.10.0
# L2CS-Net (install from github.com/Ahmednull/L2CS-Net)

# EOU detection
# livekit turn-detector (install from github.com/livekit/turn-detector)

# LLM
openai>=1.0.0
requests>=2.31.0  # For Ollama API

# Async
uvloop>=0.19.0; sys_platform != 'win32'

# Logging/metrics
psutil
```

---

## 🚀 Entry Points

```bash
# Start the full system (waits for Jackie to connect)
conda activate smait
python run_jackie.py --host 0.0.0.0 --port 8765

# Voice-only mode (no camera/vision — audio pipeline only)
python run_jackie.py --voice-only

# Test individual components
python -m smait.demo connection
python -m smait.demo audio
python -m smait.demo separation
python -m smait.demo full

# With debug overlay (show video feed with face tracking)
python run_jackie.py --debug --show-video
```

---

## 🎯 Success Criteria

- [ ] Jackie connects reliably via travel router (+ USB fallback works)
- [ ] Raw 4-channel audio AND CAE audio both streaming to PC
- [ ] DOA angles streaming to PC from CAE SDK
- [ ] Dolphin separates target speaker from 2+ simultaneous talkers
- [ ] ASR on separated audio: WER < 15% in noisy conference environment
- [ ] Proactive greeting when someone approaches + gazes for > 2s
- [ ] Correct speaker isolation (rejects bystander speech in crowd)
- [ ] Total response latency < 3s (VAD → ASR → LLM → TTS first chunk)
- [ ] Natural TTS via Kokoro played through Jackie speaker (not Android TTS)
- [ ] Clean session lifecycle: greet → converse → farewell
- [ ] Handles 10+ consecutive interactions without crashes
- [ ] Data logger captures every turn with full metrics
- [ ] Code is clean, well-documented, and defensible at HFES Q&A
- [ ] VRAM usage stays under 12GB during full operation

---

## 🏆 What Makes This Academically Novel

This system implements a **5-layer speaker isolation pipeline** — no single HRI system in the literature combines all of these:

1. **Hardware acoustic preprocessing** — iFLYTEK CAE (beamforming, AEC, noise suppression)
2. **Multi-channel spatial features** — 4-mic IPD (Inter-channel Phase Difference) for 3D sound source localization
3. **Audio-Visual Target Speaker Extraction** — Dolphin AV-TSE using discrete lip semantics + global-local attention
4. **DOA spatial verification** — Cross-checking extracted speaker against direction-of-arrival angle
5. **Semantic turn-taking** — LiveKit EOU model predicts end-of-utterance from transcript content

The contribution is the **complete integrated pipeline** running in real-time on consumer hardware (RTX 5070) for a physically embodied social robot in an uncontrolled conference environment.
