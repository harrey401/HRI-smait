# Feature Landscape

**Domain:** HRI Conference-Room Robot with Audio-Visual Speaker Separation
**Researched:** 2026-03-09 (updated with verified API findings)

## Table Stakes

Features users expect. Missing = interaction fails or feels broken.

### Audio Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| VAD-based speech segmentation | Without it, the robot cannot know when someone is speaking vs silent. Foundational to all downstream processing. | Low | Silero VAD v6.2.1 already integrated. Tune threshold per environment (0.5 default, adjust 0.3-0.7). Use 30ms chunks at 16kHz. |
| End-of-utterance detection via silence threshold | Robot must know when the speaker has finished to avoid cutting them off or waiting too long. | Medium | VAD silence duration is the proven baseline. Recommended: 500ms min silence for EOU check, 700ms auto-trigger, 1500ms hard cutoff. LiveKit EOU is unavailable -- VAD-based is the correct replacement. |
| Mic gating during TTS playback | Without echo suppression, the robot hears its own voice and enters feedback loops. | Medium | Dual-layer: server-side (AudioPipeline suppresses VAD) + app-side (mute AudioRecord). Both needed. |
| Audio-visual target speaker extraction (Dolphin) | Core value proposition. In a multi-speaker conference room, the robot MUST isolate the target speaker. | High | VERIFIED: Dolphin takes mono 16kHz audio + 88x88 grayscale lip ROIs. NOT multi-channel. Must beamform BEFORE Dolphin. Installation requires git clone (not pip). |
| Robust ASR on separated audio | Speech must be transcribed accurately. Parakeet TDT 0.6B is the right choice for edge Blackwell deployment. | Medium | Requires PyTorch nightly with CUDA 12.8+ for sm_120. CUDA graphs must be disabled (`NEMO_DISABLE_CUDA_GRAPHS=1`). |
| DOA-driven beamforming | Direction of Arrival from the 4-mic array provides spatial signal for multi-speaker disambiguation. | High | iFLYTEK CAE SDK provides DOA. Must fix 8ch->4ch mismatch via hlw.ini config. |
| TTS response (Kokoro-82M) | The robot must speak back. | Low | VERIFIED: `pip install kokoro>=0.9.4`. Use `KPipeline(lang_code='a')` generator API. Yields audio at 24kHz. Voice options: af_heart, am_adam, etc. |

### Vision Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Face detection and tracking | Robot must know where faces are. | Low | MediaPipe Face Mesh (468 landmarks) with IOU-based track persistence already implemented. |
| Lip region extraction for Dolphin | Dolphin REQUIRES mouth ROI video input (not optional). | Medium | VERIFIED: Dolphin expects 88x88 grayscale at 25fps. Current LipExtractor produces 96x96 RGB. Must preprocess: resize + grayscale conversion. Consider using MediaPipe landmarks for mouth ROI instead of Dolphin's RetinaFace dependency. |
| Gaze-based engagement detection | Robot needs to know who is looking at it. | Medium | VERIFIED: L2CS-Net from `pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main`. Pipeline API with `arch='ResNet50'`, weights auto-download from Google Drive. Pre-download recommended. |

### Session Management

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Session state machine (IDLE through DISENGAGING) | Without lifecycle management, robot behavior is random. | Medium | 5-state FSM already designed. |
| Proactive greeting on engagement | When someone looks at the robot with sustained gaze, it should greet them. | Low | Trigger on ENGAGEMENT_START after 2s gaze threshold. |
| Graceful session termination | Handle goodbye, face loss, and silence timeout without hanging. | Low | Already designed. |

### Conversation

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Multi-turn dialogue with context | Robot must remember earlier conversation. | Medium | Sliding window of last 10 turns. Phi-4 Mini Q4 via Ollama with OpenAI fallback. |
| Streaming LLM -> TTS pipeline | Response must start within 1.5s. | Medium | VERIFIED: Kokoro KPipeline generator naturally yields per-sentence. Buffer LLM tokens until sentence boundary, then call `pipeline(sentence, voice='af_heart')`. |

### System Quality

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| End-to-end latency under 1500ms | >2000ms breaks conversation flow. | High | Budget: VAD ~250ms + Dolphin ~50ms + ASR ~300ms + LLM ~400ms + TTS ~150ms = ~1150ms target. |
| VRAM budget within 12GB | All models must coexist on RTX 5070. | Medium | Budget: Parakeet ~2GB + Kokoro ~1GB + Dolphin ~0.25GB + L2CS ~0.3GB + Phi-4 Q4 ~3GB + overhead ~2GB = ~8.5GB. Fits. |
| Graceful degradation | If any model fails to load, system must still function. | Medium | Already designed. CRITICAL: Dolphin's "passthrough" fallback must return CAE mono audio, not multi-channel. |

## Differentiators

Features that set this system apart from basic HRI.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Audio-visual fusion for speaker disambiguation | Most systems use ONLY audio OR ONLY vision. Fusing DOA + lip + gaze is rare in production. | High | Dolphin handles AV extraction. Engagement detector fuses gaze + face area + DOA for target selection. |
| Walking-past filter | Distinguish passerby from approaching person. Prevents false triggers. | Low | Filter by face area trajectory + gaze duration. |
| DOA-corroborated engagement | Use DOA to confirm the visually-selected target is actually speaking. | Medium | Fuse DOA angle with face position. Flag uncertainty when they disagree. |
| Hallucination-filtered ASR | Reject phantom ASR outputs from noise. | Low | Already in Transcriber. Confidence threshold + phrase blocklist + min duration. |
| HRI checklist scoring | Quantitative evaluation of interaction quality. | Low | Already in DataLogger. Valuable for systematic improvement. |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Multi-person simultaneous conversation | Complexity explosion. Single-target-speaker is correct design. | DOA + gaze to select ONE target. Queue others via turn-taking. |
| Wake word detection | Visual engagement is more natural for a robot with a camera. | Gaze-based engagement with 2s threshold. |
| Cloud-only speech processing | Latency over WiFi to cloud blows 1500ms budget. | All ML inference on local RTX 5070. |
| Software AEC on top of hardware beamforming | Double-processing degrades audio. | Mic gating as primary echo suppression. |
| Speaker voiceprint enrollment | Adds friction. Visual target selection is enrollment-free. | Gaze + DOA dynamic selection. |
| Word-level TTS streaming | Audio artifacts from prosody breaks. | Sentence-level streaming via Kokoro KPipeline generator. |
| Complex dialogue state tracking | LLM handles context naturally through context window. | Sliding-window conversation history. |

## Feature Dependencies

```
Face Detection (MediaPipe) --> Lip Extraction (88x88 gray) --> Dolphin AV-TSE
Face Detection (MediaPipe) --> Gaze Estimation (L2CS) --> Engagement Detection
DOA (CAE Hardware) --> Engagement Detection (multi-signal fusion)
Engagement Detection --> Session State Machine --> Proactive Greeting

VAD (Silero) --> Speech Segmentation --> CAE mono audio --> Dolphin AV-TSE
Dolphin AV-TSE --> ASR (Parakeet) --> Hallucination Filter --> End-of-Utterance Detection
End-of-Utterance Detection --> Dialogue Manager (LLM)
Dialogue Manager --> TTS (Kokoro KPipeline) --> Mic Gating --> Audio Playback

Lip Extraction + Speech Segmentation --> Temporal Alignment --> Dolphin AV-TSE
  (lip frames must be time-aligned with audio segments -- critical sync point)

CAE Beamforming (Android) --> Binary WebSocket --> Server Audio Pipeline
  (8ch->4ch format fix is prerequisite for all audio features)

CRITICAL: Dolphin requires BOTH audio AND lip video. No audio-only fallback.
  If lip frames unavailable, must fall back to CAE passthrough (skip Dolphin entirely).
```

**Critical path:** CAE fix -> audio pipeline -> VAD -> lip extraction -> Dolphin -> ASR -> LLM -> TTS
**Parallel path:** Face detection -> gaze -> engagement (can develop alongside audio)
**Sync point:** Audio-visual alignment (lip frames matched to speech segments by timestamp)

## MVP Recommendation

### Must have for a working demo (Phase 1 priority):

1. **PyTorch nightly + sm_120 verification** -- everything depends on GPU
2. **CAE beamforming fix** (8ch->4ch) -- correct audio input
3. **VAD speech segmentation** -- already works, verify with real hardware
4. **Face detection + tracking** -- already works
5. **Gaze-based engagement** -- activate L2CS-Net (install from fork)
6. **Basic ASR** -- Parakeet on CAE passthrough audio
7. **Kokoro TTS** -- `KPipeline` streaming, sentence-level
8. **Mic gating** -- prevent echo loops
9. **VAD-based EOU** -- replace LiveKit with silence thresholds

### Phase 2 (full AV pipeline):

10. **Dolphin AV-TSE** -- clone repo, fix all APIs, integrate with lip extraction
11. **Lip extraction** at 88x88 grayscale for Dolphin
12. **Audio-visual temporal sync** -- timestamp alignment
13. **DOA integration** into engagement detector
14. **End-to-end latency optimization** to hit 1500ms

### Defer:

- VVAD from Dolphin's lip encoder -- nice but not essential
- HRI checklist auto-scoring -- valuable for iteration only
- User-calibrated silence thresholds -- start with fixed values

## Sources

- [JusperLee/Dolphin GitHub](https://github.com/JusperLee/Dolphin) -- verified repo structure
- [Dolphin HF Space](https://huggingface.co/spaces/JusperLee/Dolphin) -- verified API
- [Kokoro PyPI](https://pypi.org/project/kokoro/) -- v0.9.4 verified
- [hexgrad/Kokoro-82M HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) -- API, voices
- [L2CS-Net GitHub](https://github.com/Ahmednull/L2CS-Net)
- [edavalosanaya/L2CS-Net fork](https://github.com/edavalosanaya/L2CS-Net) -- pip-installable
- [Silero VAD PyPI](https://pypi.org/project/silero-vad/) -- v6.2.1
- [LiveKit VAD docs](https://docs.livekit.io/agents/logic/turns/vad/) -- silence thresholds
- [Dolphin paper](https://arxiv.org/html/2509.23610)

---

*Feature landscape analysis: 2026-03-09 (updated with verified API findings)*
