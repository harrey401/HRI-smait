# SMAIT v3 — Full Hardware Integration

## What This Is

A Human-Robot Interaction (HRI) system for the Jackie robot that enables natural, multi-turn conversation in noisy conference room environments. The system uses a 4-microphone array with beamforming, camera-based face/gaze tracking, and audio-visual ML fusion to isolate and interact with a single target speaker amid background noise. Spans two codebases: a Python edge server (SMAIT-v3) and an Android companion app (smait-jackie-app).

## Core Value

The robot must reliably isolate and converse with one person in a noisy multi-speaker conference room — using directional audio (DOA/beamforming) and visual cues (lip movement, gaze) to separate the target speaker from background chatter.

## Requirements

### Validated

<!-- Existing capabilities confirmed from codebase analysis -->

- ✓ Event-driven pipeline architecture with central EventBus (29 event types) — existing
- ✓ WebSocket binary protocol with frame type demuxing (0x01 CAE, 0x02 video, 0x03 raw, 0x05 TTS) — existing
- ✓ Silero VAD integration with ring buffer and mic gating — existing
- ✓ MediaPipe Face Mesh 468-landmark face tracking with IOU-based persistent track IDs — existing
- ✓ Dataclass-based hierarchical config system with env/file/CLI loading — existing
- ✓ Session state machine (IDLE → APPROACHING → ENGAGED → CONVERSING → DISENGAGING) — existing
- ✓ Dialogue manager with Ollama (Phi-4 Mini) + OpenAI fallback, streaming responses — existing
- ✓ Structured logging, metrics tracking, and HRI checklist scoring — existing
- ✓ Graceful degradation with fallbacks for every ML model — existing
- ✓ Android app with WebSocket client and basic AudioRecord capture — existing

### Active

<!-- Current scope: activating all hardware and ML models -->

- [ ] CAE beamforming integration in Android app (merge cae-work-march2, fix 8ch→4ch mismatch)
- [ ] Android app sends 3 streams: CAE audio (0x01), raw 4-channel audio (0x03), video (0x02)
- [ ] Android app sends DOA angles from CAE callbacks as JSON
- [ ] Android app receives and plays TTS audio (0x05 frames via AudioTrack)
- [ ] Dolphin AV-TSE model loaded and wired for audio-visual speaker extraction
- [ ] Kokoro-82M TTS replacing Piper, streaming 24kHz synthesis
- [ ] L2CS-Net gaze estimation activated (stub → real model)
- [ ] VAD-based end-of-utterance detection (replacing LiveKit EOU which is unavailable)
- [ ] Parakeet TDT ASR verified on Blackwell sm_120 GPU
- [ ] Full audio pipeline: CAE + raw audio → VAD → Dolphin AV-TSE → ASR
- [ ] Full vision pipeline: JPEG → face tracking → gaze → lip extraction → engagement
- [ ] Audio-visual sync: align lip frames with speech segments by timestamp
- [ ] DOA angles integrated into engagement detector for multi-speaker disambiguation
- [ ] Complete conversation loop: engagement → greeting → speech → separation → transcription → LLM → TTS → playback
- [ ] Turn-taking via VAD silence thresholds (Silero-based)
- [ ] TTS mic gating (mute mic during robot speech, both server and app side)
- [ ] Code cleanup: remove dead code, redundant imports, unused stubs after changes
- [ ] Unit + integration tests at 80%+ coverage
- [ ] End-to-end latency: speech end → TTS start < 1500ms
- [ ] VRAM budget stays within 12GB (target 8.5-9.5GB)

### Out of Scope

- LiveKit EOU detector — package is private/unavailable, replaced by VAD-based approach
- Multi-person simultaneous conversation — robot focuses on one target speaker at a time
- Cloud-only deployment — system runs on local edge server with GPU
- Wake word detection — robot uses visual engagement (gaze/approach) to initiate
- Piper TTS — replaced by Kokoro-82M
- Isaac Sim testing in this milestone — simulation tests are separate effort

## Context

- **Brownfield project:** Server code has full architecture with stub ML models. All event wiring exists in `smait/main.py` but models aren't loaded with real weights.
- **Android app state:** Main branch has basic AudioRecord. Branch `cae-work-march2` has CAE beamforming work that was reverted for demo stability. Needs merge + fix.
- **Hardware:** iFLYTEK RK3588 board with 4-mic linear array. USB mic reports 8 channels at 16-bit, but CAE SDK expects 4 channels — format mismatch needs resolution via `hlw.ini` config.
- **GPU:** NVIDIA RTX 5070 (Blackwell sm_120 architecture, 12GB VRAM). CUDA graphs disabled for NeMo compatibility.
- **Target environment:** Conference rooms with multiple speakers, background noise, varying lighting.
- **Spec document:** `REBUILD_PROMPT_FINAL.md` (56KB) contains the full v3 specification.

## Constraints

- **Hardware**: 12GB VRAM budget — all models must fit simultaneously (Parakeet ~2GB, Kokoro ~1GB, Dolphin ~2-3GB, L2CS ~0.3GB, Phi-4 ~3GB Q4)
- **Latency**: Speech-end to TTS-start must be < 1500ms for natural conversation
- **Network**: WiFi between robot and server, ~1MB/s bandwidth (128KB/s audio + 500KB/s video)
- **GPU Architecture**: Blackwell sm_120 requires CUDA graphs disabled, PyTorch nightly may be needed
- **Single Client**: WebSocket server accepts one robot connection at a time
- **Code Quality**: Clean code — no redundant files, dead imports, or unused stubs after changes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace LiveKit EOU with VAD-based detection | LiveKit turn-detector package is private/unavailable | — Pending |
| Keep Dolphin AV-TSE as speaker separation model | Core innovation for noisy conference room scenario | — Pending |
| Use Kokoro-82M for TTS | Already in codebase stubs, ~1GB VRAM, 24kHz streaming | — Pending |
| VAD silence thresholds for turn-taking | Simple, proven, Silero VAD already integrated | — Pending |
| DOA + beamforming as primary speaker direction signal | Critical for conference room multi-speaker disambiguation | — Pending |
| Merge CAE from cae-work-march2 branch | Contains beamforming work needed for directional audio | — Pending |

---
*Last updated: 2026-03-09 after initialization*
