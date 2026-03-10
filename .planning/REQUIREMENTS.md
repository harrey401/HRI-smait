# Requirements: SMAIT v3 Full Hardware Integration

**Defined:** 2026-03-09
**Core Value:** Robot reliably isolates and converses with one person in a noisy conference room using directional audio and visual cues

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Environment

- [ ] **ENV-01**: PyTorch nightly with CUDA 12.8+ verified on Blackwell sm_120
- [ ] **ENV-02**: All ML models load simultaneously within 12GB VRAM budget
- [x] **ENV-03**: Dolphin AV-TSE vendored from source (not pip — no setup.py exists)

### Audio Pipeline

- [ ] **AUD-01**: Android app integrates CAE beamforming (merge cae-work-march2 via revert-the-revert)
- [ ] **AUD-02**: CAE 8-channel to 4-channel format mismatch resolved via hlw.ini config
- [ ] **AUD-03**: Android app sends 3 streams: CAE audio (0x01), raw 4-channel audio (0x03), video (0x02)
- [ ] **AUD-04**: Android app sends DOA angles from CAE callbacks as JSON messages
- [ ] **AUD-05**: Silero VAD segments speech from CAE audio with ring buffer alignment
- [ ] **AUD-06**: Acoustic echo cancellation replaces mic gating (research CAE SDK AEC + software AEC, implement best approach)
- [ ] **AUD-07**: Barge-in support — robot listens while speaking, user can interrupt mid-response

### Speaker Separation

- [ ] **SEP-01**: Dolphin AV-TSE loaded with correct API (`from look2hear.models import Dolphin`)
- [ ] **SEP-02**: Audio input preprocessed to mono `[1, samples]` at 16kHz for Dolphin
- [ ] **SEP-03**: Lip frames preprocessed to 88x88 grayscale at 25fps for Dolphin
- [ ] **SEP-04**: Audio-visual temporal sync via server-side monotonic timestamps
- [ ] **SEP-05**: DOA angles integrated into engagement detector for multi-speaker disambiguation
- [ ] **SEP-06**: Fallback to CAE passthrough audio when Dolphin is unavailable

### Speech Recognition

- [ ] **ASR-01**: Parakeet TDT ASR verified on Blackwell sm_120 with CUDA graphs disabled
- [ ] **ASR-02**: Hallucination filtering rejects phantom transcripts (confidence + phrase blocklist)
- [ ] **ASR-03**: VAD-based end-of-utterance detection with ~1.8s silence threshold (replaces LiveKit EOU)

### Vision Pipeline

- [ ] **VIS-01**: L2CS-Net gaze estimation activated with correct arch (`ResNet50`, not `Gaze360`)
- [ ] **VIS-02**: Lip extraction produces mouth ROI compatible with Dolphin (88x88 grayscale from MediaPipe landmarks)
- [ ] **VIS-03**: Gaze-based engagement detection with sustained gaze threshold (>2s)
- [ ] **VIS-04**: Face tracking maintains persistent IDs across frames (existing MediaPipe + IOU)

### TTS & Playback

- [ ] **TTS-01**: Kokoro-82M TTS integrated with correct API (`KPipeline(lang_code='a')` generator)
- [ ] **TTS-02**: Sentence-level streaming TTS (yield per sentence, not per word)
- [ ] **TTS-03**: TTS audio sent as 0x05 binary frames to Android for AudioTrack playback
- [ ] **TTS-04**: Android app plays TTS audio via AudioTrack on speaker

### Conversation Loop

- [ ] **CONV-01**: Full loop: engagement -> greeting -> speech -> separation -> ASR -> LLM -> TTS -> playback
- [ ] **CONV-02**: Session state machine drives interaction lifecycle (idle -> approaching -> engaged -> conversing -> disengaging)
- [ ] **CONV-03**: Proactive greeting when sustained engagement detected
- [ ] **CONV-04**: Goodbye detection ends conversation gracefully

### Code Quality

- [ ] **QUAL-01**: All stub APIs corrected to match real model interfaces
- [ ] **QUAL-02**: Dead code removed after stub replacement (unused imports, old class references, commented-out code)
- [ ] **QUAL-03**: Unit + integration tests at 80%+ coverage
- [ ] **QUAL-04**: End-to-end latency: speech end -> TTS start < 1500ms
- [ ] **QUAL-05**: No redundant files or duplicate logic across modules

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Multi-Speaker

- **MULTI-01**: Simultaneous conversation with 2+ speakers
- **MULTI-02**: Speaker diarization and per-speaker transcript history

### Advanced Interaction

- **ADV-01**: Emotion recognition from voice prosody and facial expression
- **ADV-02**: Gesture recognition for non-verbal commands
- **ADV-03**: Multi-language support (beyond English)

### Simulation

- **SIM-01**: Isaac Sim integration testing with domain randomization
- **SIM-02**: Audio2Face ground truth for ASD validation

## Out of Scope

| Feature | Reason |
|---------|--------|
| LiveKit EOU detector | Package is private/unavailable -- replaced by VAD-based approach |
| Wake word detection | Robot uses visual engagement (gaze/approach) to initiate |
| Piper TTS | Replaced by Kokoro-82M |
| Multi-person simultaneous conversation | Robot focuses on one target speaker -- v2 feature |
| Software AEC as standalone | Will be evaluated alongside CAE SDK AEC -- best approach wins |
| Mic gating as permanent solution | Band-aid that blocks barge-in -- replaced by proper AEC |
| Speaker enrollment/voiceprint | Visual target selection is enrollment-free |
| Cloud-only deployment | System runs on local edge server with GPU |

## Traceability

| Requirement | Phase | Location | Status |
|-------------|-------|----------|--------|
| ENV-01 | Phase 7 | LAB | Pending |
| ENV-02 | Phase 7 | LAB | Pending |
| ENV-03 | Phase 1 | HOME | Pending |
| AUD-01 | Phase 6 | HOME+LAB | Pending |
| AUD-02 | Phase 6 | HOME+LAB | Pending |
| AUD-03 | Phase 6 | HOME+LAB | Pending |
| AUD-04 | Phase 6 | HOME+LAB | Pending |
| AUD-05 | Phase 4 | HOME | Pending |
| AUD-06 | Phase 5 | HOME | Pending |
| AUD-07 | Phase 5 | HOME | Pending |
| SEP-01 | Phase 4 | HOME | Pending |
| SEP-02 | Phase 4 | HOME | Pending |
| SEP-03 | Phase 4 | HOME | Pending |
| SEP-04 | Phase 4 | HOME | Pending |
| SEP-05 | Phase 4 | HOME | Pending |
| SEP-06 | Phase 4 | HOME | Pending |
| ASR-01 | Phase 7 | LAB | Pending |
| ASR-02 | Phase 5 | HOME | Pending |
| ASR-03 | Phase 5 | HOME | Pending |
| VIS-01 | Phase 3 | HOME | Pending |
| VIS-02 | Phase 3 | HOME | Pending |
| VIS-03 | Phase 3 | HOME | Pending |
| VIS-04 | Phase 3 | HOME | Pending |
| TTS-01 | Phase 2 | HOME | Pending |
| TTS-02 | Phase 2 | HOME | Pending |
| TTS-03 | Phase 2 | HOME | Pending |
| TTS-04 | Phase 6 | HOME+LAB | Pending |
| CONV-01 | Phase 8 | LAB | Pending |
| CONV-02 | Phase 8 | LAB | Pending |
| CONV-03 | Phase 8 | LAB | Pending |
| CONV-04 | Phase 8 | LAB | Pending |
| QUAL-01 | Phase 1 | HOME | Pending |
| QUAL-02 | Phase 8 | LAB | Pending |
| QUAL-03 | Phase 8 | LAB | Pending |
| QUAL-04 | Phase 8 | LAB | Pending |
| QUAL-05 | Phase 8 | LAB | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0
- HOME phases: 1-5 (can do from home)
- LAB phases: 7-8 (need RTX 5070 + robot)
- MIXED: Phase 6 (code at home, test in lab)

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-09 after HOME/LAB restructure*
