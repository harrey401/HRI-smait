# Roadmap: SMAIT v3 Full Hardware Integration

## Overview

Transform the SMAIT v3 HRI system from stub-based architecture to fully operational audio-visual ML pipeline. The journey starts by establishing a working GPU environment and fixing broken stub APIs, then builds outward through individual model integrations (TTS, vision, speaker separation), layering in conversation flow mechanics, and culminating in full end-to-end integration with quality validation. Android CAE work runs early and in parallel since it is decoupled from server-side model work.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Environment & API Foundation** - Verify sm_120 GPU, vendor Dolphin, fix all stub APIs
- [ ] **Phase 2: Android Audio Pipeline** - Merge CAE beamforming, send 3 streams + DOA from app
- [ ] **Phase 3: TTS Pipeline** - Integrate Kokoro-82M with streaming playback to Android
- [ ] **Phase 4: Vision Pipeline** - Activate gaze estimation, lip extraction, and engagement detection
- [ ] **Phase 5: Speaker Separation** - Integrate Dolphin AV-TSE with audio-visual fusion
- [ ] **Phase 6: Turn-Taking & Echo Cancellation** - VAD-based EOU, AEC research, barge-in support
- [ ] **Phase 7: Full Integration & Quality** - Wire conversation loop, optimize latency, test coverage

## Phase Details

### Phase 1: Environment & API Foundation
**Goal**: Every ML model can be imported and instantiated on the Blackwell GPU without errors
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, QUAL-01, ASR-01
**Success Criteria** (what must be TRUE):
  1. PyTorch nightly loads and reports sm_120 CUDA capability on the RTX 5070
  2. Parakeet TDT ASR runs a test transcription on the Blackwell GPU with CUDA graphs disabled
  3. Dolphin source is vendored and `from look2hear.models import Dolphin` succeeds
  4. All stub imports and class instantiations across Dolphin, Kokoro, L2CS-Net, and ASR match their real APIs (no ImportError or wrong class names)
  5. All models load simultaneously and `nvidia-smi` confirms total VRAM usage under 12GB
**Plans**: TBD

Plans:
- [ ] 01-01: TBD
- [ ] 01-02: TBD
- [ ] 01-03: TBD

### Phase 2: Android Audio Pipeline
**Goal**: The Android app captures and transmits beamformed audio, raw audio, video, and DOA angles to the server
**Depends on**: Phase 1 (server must accept streams correctly)
**Requirements**: AUD-01, AUD-02, AUD-03, AUD-04
**Success Criteria** (what must be TRUE):
  1. CAE beamforming code from cae-work-march2 is merged and the 8ch-to-4ch format mismatch is resolved in hlw.ini
  2. Android app sends three binary streams over WebSocket: CAE audio (0x01), raw 4-channel audio (0x03), and video frames (0x02)
  3. Android app sends DOA angle JSON messages from CAE callbacks, and server logs receive them
  4. Server demuxes all three streams and DOA messages without errors or dropped frames
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: TTS Pipeline
**Goal**: The robot speaks using Kokoro TTS with streaming sentence-level synthesis played back on the Android device
**Depends on**: Phase 1 (Kokoro API stubs fixed), Phase 2 (Android receives 0x05 frames)
**Requirements**: TTS-01, TTS-02, TTS-03, TTS-04
**Success Criteria** (what must be TRUE):
  1. Kokoro-82M loads via `KPipeline(lang_code='a')` and generates 24kHz audio from text
  2. TTS streams audio sentence-by-sentence (not word-by-word), with the first sentence arriving before the full response is synthesized
  3. Server sends TTS audio as 0x05 binary frames over WebSocket
  4. Android app receives 0x05 frames and plays audio through AudioTrack speaker with no clipping or gaps
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Vision Pipeline
**Goal**: The server extracts gaze direction, lip regions, and engagement state from camera video frames
**Depends on**: Phase 1 (L2CS-Net stub fixed)
**Requirements**: VIS-01, VIS-02, VIS-03, VIS-04
**Success Criteria** (what must be TRUE):
  1. L2CS-Net loads with `arch='ResNet50'` and produces yaw/pitch gaze angles per detected face
  2. Lip extraction produces 88x88 grayscale mouth ROI crops from MediaPipe landmarks, compatible with Dolphin input spec
  3. Engagement detector triggers after sustained gaze (>2 seconds) toward the robot
  4. Face tracking maintains persistent IDs across consecutive frames using existing MediaPipe + IOU tracking
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Speaker Separation
**Goal**: Dolphin AV-TSE isolates the target speaker's voice using lip video and beamformed audio
**Depends on**: Phase 1 (Dolphin vendored), Phase 4 (lip extraction ready)
**Requirements**: SEP-01, SEP-02, SEP-03, SEP-04, SEP-05, SEP-06, AUD-05
**Success Criteria** (what must be TRUE):
  1. Dolphin loads from vendored source and processes mono 16kHz audio `[1, samples]` with grayscale lip frames `[1, 1, frames, 88, 88, 1]`
  2. Audio-visual temporal sync aligns lip frames to speech segments using server-side monotonic timestamps
  3. Silero VAD segments speech from CAE audio with ring buffer, feeding segments to Dolphin
  4. DOA angles from Android are consumed by engagement detector for multi-speaker disambiguation
  5. When Dolphin is unavailable, system falls back to CAE passthrough audio without crashing
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD

### Phase 6: Turn-Taking & Echo Cancellation
**Goal**: The robot knows when the user has finished speaking and can listen while speaking (barge-in)
**Depends on**: Phase 3 (TTS active for echo cancellation testing), Phase 5 (VAD feeding pipeline)
**Requirements**: ASR-02, ASR-03, AUD-06, AUD-07
**Success Criteria** (what must be TRUE):
  1. VAD-based end-of-utterance detection triggers after ~1.8 seconds of silence, replacing LiveKit EOU
  2. ASR hallucination filtering rejects phantom transcripts using confidence thresholds and phrase blocklist
  3. Acoustic echo cancellation (CAE SDK AEC or software AEC -- best approach from research) replaces mic gating
  4. User can interrupt the robot mid-response (barge-in), causing TTS to stop and the robot to listen
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: Full Integration & Quality
**Goal**: Complete conversation loop works end-to-end with production quality and test coverage
**Depends on**: All previous phases
**Requirements**: CONV-01, CONV-02, CONV-03, CONV-04, QUAL-02, QUAL-03, QUAL-04, QUAL-05
**Success Criteria** (what must be TRUE):
  1. Full conversation loop works: engagement detection triggers greeting, user speaks, speech is separated and transcribed, LLM generates response, TTS plays back on robot
  2. Session state machine drives lifecycle correctly (idle to approaching to engaged to conversing to disengaging)
  3. Robot proactively greets when sustained engagement is detected; goodbye detection ends conversation gracefully
  4. End-to-end latency from speech end to TTS start is under 1500ms
  5. Dead code is removed, no redundant files remain, and unit + integration tests achieve 80%+ coverage
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD
- [ ] 07-03: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

Note: Phase 2 (Android) can execute in parallel with Phases 3-4 (server-side models) since they are on different codebases.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment & API Foundation | 0/3 | Not started | - |
| 2. Android Audio Pipeline | 0/2 | Not started | - |
| 3. TTS Pipeline | 0/2 | Not started | - |
| 4. Vision Pipeline | 0/2 | Not started | - |
| 5. Speaker Separation | 0/3 | Not started | - |
| 6. Turn-Taking & Echo Cancellation | 0/2 | Not started | - |
| 7. Full Integration & Quality | 0/3 | Not started | - |
