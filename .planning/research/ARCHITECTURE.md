# Architecture Patterns

**Domain:** Audio-Visual HRI ML Pipeline Integration
**Researched:** 2026-03-09 (updated with verified API findings)

## Recommended Architecture

The SMAIT v3 system has a well-designed event-driven architecture. The challenge is wiring real ML models into existing stubs while managing VRAM, synchronization, and fallback integrity. This document describes component boundaries, data flows, and integration patterns with VERIFIED model APIs.

### System Overview

```
Jackie (Android)                          Edge Server (RTX 5070, 12GB VRAM)
+---------------------+     WebSocket     +------------------------------------------------+
| CAE SDK (4-mic)     | ---0x01 audio---> | ConnectionManager                              |
| AudioRecord (raw)   | ---0x03 audio---> |    |                                            |
| Camera (JPEG)       | ---0x02 video---> |    v                                            |
| DOA callback        | ---JSON DOA-----> | EventBus (29 event types)                      |
|                     |                   |    |                                            |
|                     | <--0x05 TTS------ |    +-> Audio Pipeline  -> Separation -> ASR     |
|                     | <--JSON state---- |    +-> Vision Pipeline -> Engagement -> Dolphin |
|                     | <--JSON text----- |    +-> Dialogue -> TTS -> Connection            |
+---------------------+                   +------------------------------------------------+
```

### Component Boundaries (Updated with Verified APIs)

| Component | Responsibility | Key API | GPU? | VRAM |
|-----------|---------------|---------|------|------|
| **ConnectionManager** | WebSocket server, frame demux | Existing | No | 0 |
| **AudioPipeline** | Silero VAD, ring buffer, mic gating | `silero_vad.load_silero_vad()` | CPU | ~50MB |
| **VideoPipeline** | JPEG decode, frame buffer | Existing | No | 0 |
| **FaceTracker** | MediaPipe Face Mesh, persistent track IDs | Existing | CPU | 0 |
| **LipExtractor** | Mouth ROI crop, per-face temporal buffer | Must output 88x88 gray for Dolphin | No | 0 |
| **GazeEstimator** | L2CS-Net gaze direction | `l2cs.Pipeline(arch='ResNet50', device=cuda)` | GPU | ~300MB |
| **EngagementDetector** | Multi-signal engagement FSM | Existing | No | 0 |
| **DolphinSeparator** | AV-TSE target speaker extraction | `look2hear.models.Dolphin.from_pretrained("JusperLee/Dolphin")` | GPU | ~250MB + ~500MB runtime |
| **ParakeetASR** | NeMo speech-to-text | `nemo_toolkit[asr]` with `NEMO_DISABLE_CUDA_GRAPHS=1` | GPU | ~2GB |
| **EOUDetector** | End-of-utterance detection (VAD-based) | Heuristic + silence thresholds (no LiveKit) | CPU | 0 |
| **DialogueManager** | LLM response (Ollama/OpenAI) | Existing | GPU (Ollama) | ~3GB (Phi-4 Q4) |
| **TTSEngine** | Kokoro-82M speech synthesis | `kokoro.KPipeline(lang_code='a')` generator | GPU | ~1GB |
| **SessionManager** | Interaction lifecycle FSM | Existing | No | 0 |

### Data Flow

#### Primary Audio Pipeline (Speech to Response)

Critical latency path. Target: speech-end to TTS-start < 1500ms.

```
1. Jackie sends CAE audio (0x01) + raw 4ch audio (0x03) simultaneously
   |
2. ConnectionManager demuxes by type byte, emits events
   |
3. AudioPipeline:
   - CAE mono audio -> Silero VAD (30ms chunks, speech probability)
   - On silence after speech -> emit SpeechSegment
     (contains CAE mono audio + timestamps)
   |
4. HRISystem.on_speech_segment():
   - Query LipExtractor.get_lip_frames(target_track_id, start_time, end_time)
   - IF lip_frames available:
       Preprocess: resize to 88x88, convert grayscale
       audio_tensor = mono_audio[None]                        # [1, samples]
       video_tensor = lip_frames[None, None].float()          # [1, 1, frames, 88, 88, 1]
       est_sources = dolphin_model(audio_tensor, video_tensor) # [1, samples]
   - ELSE (no lip frames):
       Use CAE audio passthrough (skip Dolphin entirely)
   |
5. Transcriber.process_separated_audio() -> Parakeet ASR
   - Hallucination filter + confidence check
   - Update EOUDetector with transcript
   |
6. EOUDetector.on_silence() evaluates end-of-turn:
   - 500ms silence: check heuristic P(eou) from punctuation
   - 700ms silence: auto-trigger END_OF_TURN
   - 1500ms silence: hard cutoff
   |
7. DialogueManager.ask_streaming() -> LLM -> token stream
   |
8. TTSEngine:
   - Buffer tokens until sentence boundary
   - pipeline = KPipeline(lang_code='a')
   - for gs, ps, audio in pipeline(sentence, voice='af_heart'):
       Convert float32 -> int16 PCM
       Emit TTS_AUDIO_CHUNK
   |
9. ConnectionManager sends 0x05 frames to Jackie for playback
```

**CRITICAL CHANGE from stubs:** Dolphin takes mono audio, NOT multi-channel. The raw 4-channel audio is NOT passed to Dolphin. CAE beamformed mono output is the input to both VAD and Dolphin.

#### Vision Pipeline (Engagement to Target Lock)

Runs at ~30 FPS in parallel with audio. Feeds lip frames to Dolphin.

```
1. Jackie sends JPEG frames (0x02) at ~15-30 FPS
   |
2. VideoPipeline.process_jpeg() -> BGR numpy array
   |
3. Video loop:
   a. FaceTracker.process_frame() -> FaceTrack[] with persistent IDs
   b. For each track:
      - GazeEstimator: gaze_pipeline.step(face_crop) -> yaw, pitch
      - LipExtractor: crop mouth ROI -> buffer per face (88x88 grayscale at 25fps)
   c. EngagementDetector.update(tracks, gaze_results, timestamp)
   |
4. On sustained gaze > 2s:
   - Emit ENGAGEMENT_START with target_track_id
   - SessionManager transitions IDLE -> ENGAGED
   |
5. LipExtractor buffer feeds Dolphin when speech segment arrives
```

#### Verified Dolphin Inference Detail

From the HuggingFace Space Inference.py:

```python
# Installation (not pip installable)
# git clone https://github.com/JusperLee/Dolphin.git
# cd Dolphin && pip install -r requirements.txt
# Add repo root to PYTHONPATH

# Model loading
from look2hear.models import Dolphin
model = Dolphin.from_pretrained("JusperLee/Dolphin").to(device)
model.eval()

# Inference (4-second sliding window with overlap-add)
# Audio: mono [1, samples] at 16kHz
audio_tensor = mono_audio[None].to(device)

# Video: [1, 1, num_frames, 88, 88, 1] grayscale lip ROIs
# Mouth ROIs preprocessed via look2hear's get_preprocessing_pipelines()["val"]
video_tensor = torch.from_numpy(mouth_roi[None, None]).float().to(device)

# Forward pass
with torch.no_grad():
    est_sources = model(audio_tensor, video_tensor)
# Output: [1, samples] separated audio at 16kHz
```

**Window strategy:** 4-second windows with Hann overlap-add reconstruction. For real-time: process most recent 2-4 seconds on each VAD speech-end event.

#### Verified Kokoro TTS Detail

```python
# Installation: pip install kokoro>=0.9.4 soundfile
# Also: sudo apt install espeak-ng

from kokoro import KPipeline
import numpy as np

pipeline = KPipeline(lang_code='a')  # 'a' = American English

# Generation (yields sentence-level chunks)
for graphemes, phonemes, audio in pipeline(text, voice='af_heart', speed=1.0):
    # audio: numpy float32 at 24kHz
    pcm_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()
    event_bus.emit(EventType.TTS_AUDIO_CHUNK, {"audio": pcm_bytes})

# Available voices:
# American Female: af_heart, af_bella, af_nicole, af_sarah, af_sky
# American Male: am_adam, am_michael
# British Female: bf_emma, bf_isabella
# British Male: bm_george, bm_lewis
```

#### Verified L2CS-Net Detail

```python
# Installation: pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main

from l2cs import Pipeline as L2CSPipeline

gaze_pipeline = L2CSPipeline(
    weights=None,           # Auto-download (or local path)
    arch='ResNet50',        # NOT 'Gaze360' as stub says
    device=torch.device('cuda')
)

# Inference
results = gaze_pipeline.step(face_crop_bgr)  # BGR numpy array
if results and hasattr(results, 'yaw') and len(results.yaw) > 0:
    yaw = float(results.yaw[0])    # degrees
    pitch = float(results.pitch[0])  # degrees
```

## Patterns to Follow

### Pattern 1: Feature Flag Model Loading (keep existing pattern)

Already proven in the codebase. All models have `_available` flags with graceful fallback.

**Integration order:** Wire passthrough first, verify pipeline end-to-end, then swap in real model. Never wire model and pipeline simultaneously.

### Pattern 2: Timestamp-Anchored Cross-Modal Buffers

Use monotonic timestamps as universal sync primitive between audio and video.

```python
# LipExtractor maintains per-face temporal buffer
self._buffers[track_id].append(LipROI(image=gray_88x88, timestamp=timestamp))

# When speech segment arrives, extract corresponding lip frames
lip_frames = self.lip_extractor.get_lip_frames(
    target_track_id, segment.start_time, segment.end_time
)
```

Both audio and video timestamps must come from same monotonic clock (server-side `time.monotonic()`).

### Pattern 3: Kokoro Generator -> Event Stream Adapter

The `KPipeline` generator pattern maps directly to the event-driven TTS architecture:

```python
async def speak_streaming(self, text_generator):
    self._event_bus.emit(EventType.TTS_START)
    try:
        text_buffer = ""
        async for chunk in text_generator:
            text_buffer += chunk
            while SENTENCE_BOUNDARY.search(text_buffer):
                sentence, text_buffer = split_at_boundary(text_buffer)
                # KPipeline yields per-sentence audio
                for gs, ps, audio in self._pipeline(sentence, voice=self._voice):
                    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                    self._event_bus.emit(EventType.TTS_AUDIO_CHUNK, {"audio": pcm.tobytes()})
        # Flush remaining
        if text_buffer.strip():
            for gs, ps, audio in self._pipeline(text_buffer.strip(), voice=self._voice):
                pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                self._event_bus.emit(EventType.TTS_AUDIO_CHUNK, {"audio": pcm.tobytes()})
    finally:
        self._event_bus.emit(EventType.TTS_END)
```

### Pattern 4: Dolphin Preprocessing Adapter

Dolphin expects specific input formats. Adapt in DolphinSeparator, not in upstream components:

```python
def _preprocess_lip_frames(self, lip_rois: list[LipROI]) -> torch.Tensor:
    """Convert LipExtractor output to Dolphin input format."""
    frames = []
    for roi in lip_rois:
        gray = cv2.cvtColor(roi.image, cv2.COLOR_RGB2GRAY) if roi.image.ndim == 3 else roi.image
        resized = cv2.resize(gray, (88, 88))
        frames.append(resized)
    arr = np.stack(frames, axis=0)  # [T, 88, 88]
    arr = arr[..., np.newaxis]       # [T, 88, 88, 1]
    # Apply look2hear val preprocessing if needed
    return torch.from_numpy(arr[np.newaxis, np.newaxis]).float()  # [1, 1, T, 88, 88, 1]
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Passing Multi-Channel Audio to Dolphin
**What:** Sending raw 4-channel audio to Dolphin as the stub does.
**Why bad:** Dolphin takes mono only. Multi-channel input causes shape mismatch or silent corruption.
**Instead:** CAE beamform -> mono 16kHz -> Dolphin.

### Anti-Pattern 2: Blocking Event Loop with GPU Inference
**What:** Running long GPU operations (ASR 200-400ms, LLM) directly in async handlers.
**Why bad:** Blocks all event processing. Video frames pile up.
**Instead:** Use `asyncio.to_thread()` for operations > 50ms.

### Anti-Pattern 3: Assuming Dolphin Has Audio-Only Mode
**What:** Calling `model(audio_tensor)` without video, as stub does with `lip_tensor = None`.
**Why bad:** Dolphin requires both audio and video tensors. No audio-only mode verified.
**Instead:** If no lip frames available, skip Dolphin entirely and use CAE passthrough.

### Anti-Pattern 4: Using Stub Class Names in New Code
**What:** Writing code that imports `KokoroTTS`, `DolphinModel`, or `EOUModel`.
**Why bad:** These classes do not exist. Real classes are `KPipeline`, `Dolphin`, and no LiveKit model.
**Instead:** Use verified imports from STACK.md.

### Anti-Pattern 5: Loading All Models Simultaneously at Startup
**What:** Loading 5+ models at once.
**Why bad:** Peak VRAM during loading + fragmentation.
**Instead:** Load in order (largest first), log VRAM after each, use `expandable_segments:True`.

## VRAM Budget (Updated)

| Model | VRAM | Loading Strategy |
|-------|------|-----------------|
| Silero VAD | ~50 MB | CPU, load first |
| L2CS-Net | ~300 MB | GPU, load early |
| Dolphin AV-TSE | ~250 MB model + ~500 MB runtime | GPU, load after L2CS |
| Parakeet TDT 0.6B | ~2.0 GB | GPU, load before Kokoro |
| Kokoro-82M | ~1.0 GB | GPU, load after Parakeet |
| Phi-4 Mini Q4 | ~3.0 GB | GPU (Ollama manages) |
| PyTorch/CUDA overhead | ~1.0 GB | Unavoidable |
| **Total** | **~8.1 GB** | **12GB - 8.1GB = 3.9GB headroom** |

## Suggested Build Order

1. **Environment Setup:** PyTorch nightly + CUDA 12.8 + sm_120 verification
2. **Pipeline Verification:** End-to-end event flow with stubs
3. **Vision Pipeline:** Face tracking + L2CS-Net gaze + lip extraction (88x88 gray)
4. **Kokoro TTS:** Replace stub with KPipeline, verify streaming
5. **Dolphin AV-TSE:** Clone repo, fix imports, integrate with lip pipeline
6. **Parakeet ASR:** Verify on sm_120 with CUDA graphs disabled
7. **VAD EOU:** Replace LiveKit with silence thresholds
8. **Android CAE:** Merge branch, fix 8ch->4ch, enable DOA
9. **Integration + Tuning:** Full loop, latency optimization, threshold tuning

## Sources

- [JusperLee/Dolphin GitHub](https://github.com/JusperLee/Dolphin) -- repo structure
- [Dolphin HF Space Inference.py](https://huggingface.co/spaces/JusperLee/Dolphin) -- verified model API
- [JusperLee/Dolphin HuggingFace](https://huggingface.co/JusperLee/Dolphin) -- model weights
- [Kokoro PyPI v0.9.4](https://pypi.org/project/kokoro/) -- verified API
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) -- voices, usage
- [edavalosanaya/L2CS-Net](https://github.com/edavalosanaya/L2CS-Net) -- pip-installable fork
- [Silero VAD v6.2.1](https://pypi.org/project/silero-vad/)
- [PyTorch sm_120 support](https://github.com/pytorch/pytorch/issues/164342)

---

*Architecture analysis: 2026-03-09 (updated with verified API findings)*
