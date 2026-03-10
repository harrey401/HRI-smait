# Roadmap: SMAIT v3 Full Hardware Integration

## Overview

Transform the SMAIT v3 HRI system from stub-based architecture to fully operational audio-visual ML pipeline. The roadmap is split into HOME phases (code writing, stub fixes, unit tests with mocked models) and LAB phases (GPU validation, hardware testing, E2E integration). This lets you maximize progress from home, then validate everything in focused lab sessions.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

### HOME phases (no GPU/robot needed):
- [x] **Phase 1: Dependency Setup & Stub API Fixes** - Vendor Dolphin, fix all stub APIs, install packages (completed 2026-03-10)
- [x] **Phase 2: TTS Pipeline Code** - Rewrite Kokoro integration with correct KPipeline API (completed 2026-03-10)
- [x] **Phase 3: Vision Pipeline Code** - Rewrite L2CS-Net gaze, lip extraction for Dolphin format (completed 2026-03-10)
- [ ] **Phase 4: Speaker Separation Code** - Rewrite Dolphin separator with correct API and tensor shapes
- [ ] **Phase 5: Turn-Taking & AEC Code** - VAD-based EOU, AEC research, barge-in logic
- [ ] **Phase 6: Android Audio Pipeline** - Merge CAE beamforming, 3 streams + DOA, AudioTrack playback

### LAB phases (RTX 5070 + robot required):
- [ ] **Phase 7: GPU Validation & Model Loading** - Verify all models on sm_120, VRAM budget, Parakeet ASR
- [ ] **Phase 8: Full Integration & Quality** - End-to-end conversation loop, latency tuning, 80% coverage

## Phase Details

### Phase 1: Dependency Setup & Stub API Fixes
**Goal**: All ML model code uses correct imports, classes, and tensor shapes — ready for GPU testing
**Location**: HOME
**Depends on**: Nothing (first phase)
**Requirements**: ENV-03, QUAL-01
**Success Criteria** (what must be TRUE):
  1. Dolphin source is vendored into the project and `from look2hear.models import Dolphin` succeeds (CPU import, no GPU needed)
  2. All stub files corrected: dolphin_separator.py, tts.py, gaze.py, eou_detector.py use real API signatures
  3. Kokoro installed via `pip install kokoro>=0.9.4`
  4. L2CS-Net installed from maintained fork
  5. Unit tests with mocked models pass for all corrected stubs

**Plans:** 3/3 plans complete

Plans:
- [ ] 01-01-PLAN.md — Vendor Dolphin, install dependencies, bootstrap test infrastructure, fix EventBus
- [ ] 01-02-PLAN.md — Fix DolphinSeparator and TTSEngine stub imports and API usage
- [ ] 01-03-PLAN.md — Fix GazeEstimator arch param and strip LiveKit from EOUDetector

### Phase 2: TTS Pipeline Code
**Goal**: Kokoro TTS wrapper rewritten with correct KPipeline API and sentence-level streaming
**Location**: HOME
**Depends on**: Phase 1 (Kokoro installed, stub fixed)
**Requirements**: TTS-01, TTS-02, TTS-03
**Success Criteria** (what must be TRUE):
  1. TTSEngine uses `KPipeline(lang_code='a')` generator API, yielding `(graphemes, phonemes, audio)` tuples
  2. Sentence-level streaming works: first sentence audio emitted before full response is synthesized
  3. TTS audio encoded as 0x05 binary frames for WebSocket transmission
  4. Unit tests verify streaming behavior with mocked KPipeline

**Plans:** 2/2 plans complete

Plans:
- [ ] 02-01-PLAN.md — Fix TTSEngine emit_async streaming, GPU-safe audio conversion, streaming tests
- [ ] 02-02-PLAN.md — Protocol and ConnectionManager TTS audio forwarding tests

### Phase 3: Vision Pipeline Code
**Goal**: Fix lip_roi_size config for Dolphin compatibility and write comprehensive unit tests for all four vision modules
**Location**: HOME
**Depends on**: Phase 1 (L2CS-Net installed, stub fixed)
**Requirements**: VIS-01, VIS-02, VIS-03, VIS-04
**Success Criteria** (what must be TRUE):
  1. GazeEstimator uses `arch='ResNet50'` (not `Gaze360`) with correct L2CS Pipeline API
  2. LipExtractor produces 88x88 grayscale mouth ROI from MediaPipe landmarks (Dolphin-compatible)
  3. Engagement detector logic uses gaze duration (>2s) + face area + DOA angles
  4. Unit tests verify lip frame output shape, gaze angle format, and engagement thresholds with mocked inputs

**Plans:** 2/2 plans complete

Plans:
- [ ] 03-01-PLAN.md — Fix lip_roi_size config to 88x88, write LipExtractor and GazeEstimator L2CS tests
- [ ] 03-02-PLAN.md — Write EngagementDetector state machine and FaceTracker IOU tests

### Phase 4: Speaker Separation Code
**Goal**: Dolphin separator rewritten with correct look2hear API, mono audio input, grayscale lip video input
**Location**: HOME
**Depends on**: Phase 1 (Dolphin vendored), Phase 3 (lip extraction format defined)
**Requirements**: SEP-01, SEP-02, SEP-03, SEP-04, SEP-05, SEP-06, AUD-05
**Success Criteria** (what must be TRUE):
  1. DolphinSeparator uses `from look2hear.models import Dolphin` with correct tensor shapes: audio `[1, samples]` mono 16kHz, video `[1, 1, frames, 88, 88, 1]` grayscale
  2. Audio-visual temporal sync logic aligns lip frames to speech segments via monotonic timestamps
  3. VAD speech segmentation feeds correctly sized segments to Dolphin
  4. DOA angle integration in engagement detector for multi-speaker disambiguation
  5. Fallback to CAE passthrough when Dolphin unavailable
  6. Unit tests verify tensor shapes, sync logic, and fallback behavior with mocked Dolphin model

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Turn-Taking & AEC Code
**Goal**: VAD-based end-of-utterance replaces LiveKit EOU, AEC approach researched and coded, barge-in logic written
**Location**: HOME
**Depends on**: Phase 2 (TTS code ready for echo testing logic)
**Requirements**: ASR-02, ASR-03, AUD-06, AUD-07
**Success Criteria** (what must be TRUE):
  1. EOUDetector rewritten to use Silero VAD silence thresholds (~1.8s) instead of LiveKit model
  2. ASR hallucination filtering logic implemented (confidence threshold + phrase blocklist)
  3. AEC approach researched and documented: CAE SDK AEC capabilities vs software AEC (speexdsp/WebRTC)
  4. Barge-in logic written: detect user speech during TTS, stop TTS, resume listening
  5. Unit tests verify EOU timing, hallucination rejection, and barge-in state transitions

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Android Audio Pipeline
**Goal**: Android app code updated for CAE beamforming, 3 streams, DOA, and AudioTrack TTS playback
**Location**: HOME (code writing) + LAB (hardware testing)
**Depends on**: Phase 1 (server protocol verified)
**Requirements**: AUD-01, AUD-02, AUD-03, AUD-04, TTS-04
**Success Criteria** (what must be TRUE):
  1. CAE beamforming merged from cae-work-march2 via revert-the-revert, 8ch-to-4ch resolved in hlw.ini
  2. App sends three binary streams: CAE audio (0x01), raw 4-channel audio (0x03), video (0x02)
  3. App sends DOA angle JSON from CAE callbacks
  4. App receives 0x05 TTS frames and plays via AudioTrack
  5. Server demuxes all streams and DOA without errors (verified in lab)

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: GPU Validation & Model Loading
**Goal**: All models verified on RTX 5070 Blackwell sm_120, VRAM budget confirmed, Parakeet ASR tested
**Location**: LAB
**Depends on**: Phases 1-5 (all code written and unit-tested)
**Requirements**: ENV-01, ENV-02, ASR-01
**Success Criteria** (what must be TRUE):
  1. PyTorch nightly with CUDA 12.8+ loads and reports sm_120 capability
  2. Parakeet TDT ASR transcribes test audio on Blackwell with CUDA graphs disabled
  3. All models (Dolphin, Kokoro, L2CS-Net, Parakeet, Silero VAD) load simultaneously
  4. `nvidia-smi` confirms total VRAM under 12GB
  5. Each model produces correct output on real GPU (not just mocked)

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: Full Integration & Quality
**Goal**: Complete conversation loop works end-to-end with production quality and test coverage
**Location**: LAB
**Depends on**: All previous phases
**Requirements**: CONV-01, CONV-02, CONV-03, CONV-04, QUAL-02, QUAL-03, QUAL-04, QUAL-05
**Success Criteria** (what must be TRUE):
  1. Full conversation loop: engagement -> greeting -> speech -> separation -> ASR -> LLM -> TTS -> playback
  2. Session state machine drives lifecycle (idle -> approaching -> engaged -> conversing -> disengaging)
  3. Proactive greeting on sustained engagement; goodbye detection ends conversation
  4. End-to-end latency speech end -> TTS start < 1500ms
  5. Dead code removed, no redundant files, unit + integration tests at 80%+ coverage

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD
- [ ] 08-03: TBD

## Progress

**Execution Order:**
- HOME: Phases 1 -> 2, 3 (parallel) -> 4 -> 5 -> 6 (code only)
- LAB: Phase 6 (hardware test) -> 7 -> 8

Phase 2 (TTS) and Phase 3 (Vision) can run in parallel since they're independent.
Phase 6 (Android) code can be written alongside Phases 2-5, but hardware testing needs the lab.

| Phase | Location | Plans Complete | Status | Completed |
|-------|----------|----------------|--------|-----------|
| 1. Dependency Setup & Stub API Fixes | 3/3 | Complete   | 2026-03-10 | - |
| 2. TTS Pipeline Code | 2/2 | Complete   | 2026-03-10 | - |
| 3. Vision Pipeline Code | 2/2 | Complete   | 2026-03-10 | - |
| 4. Speaker Separation Code | HOME | 0/3 | Not started | - |
| 5. Turn-Taking & AEC Code | HOME | 0/2 | Not started | - |
| 6. Android Audio Pipeline | HOME+LAB | 0/2 | Not started | - |
| 7. GPU Validation & Model Loading | LAB | 0/2 | Not started | - |
| 8. Full Integration & Quality | LAB | 0/3 | Not started | - |
