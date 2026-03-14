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

- [x] **AUD-01**: Android app integrates CAE beamforming (merge cae-work-march2 via revert-the-revert)
- [x] **AUD-02**: CAE 8-channel to 4-channel format mismatch resolved via hlw.ini config
- [x] **AUD-03**: Android app sends 3 streams: CAE audio (0x01), raw 4-channel audio (0x03), video (0x02)
- [x] **AUD-04**: Android app sends DOA angles from CAE callbacks as JSON messages
- [x] **AUD-05**: Silero VAD segments speech from CAE audio with ring buffer alignment
- [x] **AUD-06**: Acoustic echo cancellation replaces mic gating (research CAE SDK AEC + software AEC, implement best approach)
- [x] **AUD-07**: Barge-in support — robot listens while speaking, user can interrupt mid-response

### Speaker Separation

- [x] **SEP-01**: Dolphin AV-TSE loaded with correct API (`from look2hear.models import Dolphin`)
- [x] **SEP-02**: Audio input preprocessed to mono `[1, samples]` at 16kHz for Dolphin
- [x] **SEP-03**: Lip frames preprocessed to 88x88 grayscale at 25fps for Dolphin
- [x] **SEP-04**: Audio-visual temporal sync via server-side monotonic timestamps
- [x] **SEP-05**: DOA angles integrated into engagement detector for multi-speaker disambiguation
- [x] **SEP-06**: Fallback to CAE passthrough audio when Dolphin is unavailable

### Speech Recognition

- [ ] **ASR-01**: Parakeet TDT ASR verified on Blackwell sm_120 with CUDA graphs disabled
- [x] **ASR-02**: Hallucination filtering rejects phantom transcripts (confidence + phrase blocklist)
- [x] **ASR-03**: VAD-based end-of-utterance detection with ~1.8s silence threshold (replaces LiveKit EOU)

### Vision Pipeline

- [x] **VIS-01**: L2CS-Net gaze estimation activated with correct arch (`ResNet50`, not `Gaze360`)
- [x] **VIS-02**: Lip extraction produces mouth ROI compatible with Dolphin (88x88 grayscale from MediaPipe landmarks)
- [x] **VIS-03**: Gaze-based engagement detection with sustained gaze threshold (>2s)
- [x] **VIS-04**: Face tracking maintains persistent IDs across frames (existing MediaPipe + IOU)

### TTS & Playback

- [x] **TTS-01**: Kokoro-82M TTS integrated with correct API (`KPipeline(lang_code='a')` generator)
- [x] **TTS-02**: Sentence-level streaming TTS (yield per sentence, not per word)
- [x] **TTS-03**: TTS audio sent as 0x05 binary frames to Android for AudioTrack playback
- [x] **TTS-04**: Android app plays TTS audio via AudioTrack on speaker

### Conversation Loop

- [ ] **CONV-01**: Full loop: engagement -> greeting -> speech -> separation -> ASR -> LLM -> TTS -> playback
- [ ] **CONV-02**: Session state machine drives interaction lifecycle (idle -> approaching -> engaged -> conversing -> disengaging)
- [ ] **CONV-03**: Proactive greeting when sustained engagement detected
- [ ] **CONV-04**: Goodbye detection ends conversation gracefully

### Code Quality

- [x] **QUAL-01**: All stub APIs corrected to match real model interfaces
- [ ] **QUAL-02**: Dead code removed after stub replacement (unused imports, old class references, commented-out code)
- [ ] **QUAL-03**: Unit + integration tests at 80%+ coverage
- [ ] **QUAL-04**: End-to-end latency: speech end -> TTS start < 1500ms
- [ ] **QUAL-05**: No redundant files or duplicate logic across modules

## v2.0 Requirements — Navigation & Wayfinding

Requirements for milestone v2.0. Continues from v1.0 phases.

### Chassis Connection

- [x] **CHAS-01**: SMAIT server connects to chassis WebSocket at configurable IP/port
- [x] **CHAS-02**: Server subscribes to robot pose (x, y, theta) at regular intervals
- [x] **CHAS-03**: Server subscribes to navigation status (running/success/failed/cancelled)
- [x] **CHAS-04**: Server retrieves robot global state (battery, nav_status, velocity, control_state)
- [x] **CHAS-05**: Server can send soft e-stop command to chassis
- [x] **CHAS-06**: Connection auto-reconnects on disconnect with exponential backoff

### Map & Localization

- [x] **MAP-01**: Server retrieves LIDAR occupancy grid map as PNG from chassis
- [x] **MAP-02**: Server retrieves list of available maps/buildings/floors
- [x] **MAP-03**: Server can switch active map (building + floor)
- [x] **MAP-04**: Map image rendered with robot position overlay and sent to Jackie touchscreen

### POI Management

- [x] **POI-01**: Server retrieves current marker points (names, positions, types) from chassis
- [x] **POI-02**: Server can add current robot position as a named marker point
- [x] **POI-03**: Server can remove a marker point by name
- [x] **POI-04**: Location knowledge base maps human-friendly names to POI names ("room ENG192" → "eng192")

### Navigation

- [x] **NAV-01**: Server sends navigate-to-POI command by name via /poi service
- [x] **NAV-02**: Server subscribes to planned path and renders on map
- [x] **NAV-03**: Server monitors navigation progress and reports arrival/failure verbally
- [x] **NAV-04**: Server can cancel active navigation
- [x] **NAV-05**: Server can calculate distance between two points before navigating

### Wayfinding (LLM Integration)

- [x] **WAY-01**: LLM has tool/function for querying location database ("where is X?")
- [x] **WAY-02**: LLM has tool/function for initiating navigation ("take me to X")
- [x] **WAY-03**: When user asks "where is X?", system shows map with highlighted path on touchscreen + gives verbal directions
- [x] **WAY-04**: When user says "take me to X", robot navigates and converses during transit
- [x] **WAY-05**: Robot verbally confirms arrival or explains navigation failure

### Display & UI

- [x] **DISP-01**: Map with robot position, POIs, and path rendered as image and sent to Jackie touchscreen via WebSocket
- [x] **DISP-02**: Navigation status shown on touchscreen (navigating to X, arrived)

### Setup & Deployment

- [x] **SETUP-01**: New location setup documented: map with Deployment Tool → label POIs via voice or config
- [x] **SETUP-02**: POI config stored as JSON file per building/floor for easy portability
- [x] **SETUP-03**: System auto-detects available maps and active floor on startup

### Android App Rebuild

- [x] **APP-01**: Complete app rewrite with modern Android architecture (Jetpack Compose, MVVM)
- [x] **APP-02**: Base design system with theming support (colors, fonts, logos swappable per event)
- [x] **APP-03**: Home screen with event-customizable cards (Guided Tour, Ask Me Anything, Facilities, etc.)
- [x] **APP-04**: Navigation screen showing live map, robot position, path, and destination
- [ ] **APP-05**: Conversation UI with transcript display, mic indicator, and robot avatar
- [x] **APP-06**: Facilities/wayfinding screen — searchable list of POIs with "take me there" action
- [x] **APP-07**: Event info screen (schedule, speakers, venue map) — content loaded from config
- [x] **APP-08**: All existing audio/video/TTS WebSocket streams preserved in new app architecture
- [x] **APP-09**: Event theming via JSON config file (colors, logo, event name, cards, POI labels)

### WiE Event Customization

- [x] **WIE-01**: WiE 2026 theme applied: purple/teal/orange palette, "Engineering Beyond Imagination" branding
- [x] **WIE-02**: WiE-specific cards: session tracks, career panels, keynote info, venue directions
- [ ] **WIE-03**: Student Union POIs labeled for WiE (registration, keynote hall, panel rooms, networking area)

## Future Requirements

Deferred to future milestone.

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

### Navigation Advanced

- **NAVADV-01**: Multi-floor navigation with elevator integration
- **NAVADV-02**: Patrol mode: robot autonomously visits multiple POIs on schedule
- **NAVADV-03**: Dynamic obstacle reporting via LIDAR point cloud visualization
- **NAVADV-04**: Voice-based POI labeling ("remember this spot as the bathroom")

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
| Custom SLAM implementation | Chassis has built-in LIDAR SLAM — use via WebSocket API |
| ROS 2 integration | Chassis uses proprietary WebSocket protocol, not ROS topics |
| Manual joystick control from app | Use Deployment Tool for manual control; SMAIT focuses on autonomous nav |
| Multi-robot coordination | Single robot deployment for now |

## Traceability

| Requirement | Phase | Location | Status |
|-------------|-------|----------|--------|
| ENV-01 | Phase 7 | LAB | Pending |
| ENV-02 | Phase 7 | LAB | Pending |
| ENV-03 | Phase 1 | HOME | Complete |
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
| QUAL-01 | Phase 1 | HOME | Complete |
| QUAL-02 | Phase 8 | LAB | Pending |
| QUAL-03 | Phase 8 | LAB | Pending |
| QUAL-04 | Phase 8 | LAB | Pending |
| QUAL-05 | Phase 8 | LAB | Pending |
| CHAS-01 | Phase 9 | HOME | Pending |
| CHAS-02 | Phase 9 | HOME | Pending |
| CHAS-03 | Phase 9 | HOME | Pending |
| CHAS-04 | Phase 9 | HOME | Pending |
| CHAS-05 | Phase 9 | HOME | Pending |
| CHAS-06 | Phase 9 | HOME | Pending |
| MAP-01 | Phase 10 | HOME | Complete |
| MAP-02 | Phase 10 | HOME | Complete |
| MAP-03 | Phase 10 | HOME | Complete |
| MAP-04 | Phase 10 | HOME | Complete |
| POI-01 | Phase 10 | HOME | Pending |
| POI-02 | Phase 10 | HOME | Pending |
| POI-03 | Phase 10 | HOME | Pending |
| POI-04 | Phase 10 | HOME | Pending |
| NAV-01 | Phase 10 | HOME | Pending |
| NAV-02 | Phase 10 | HOME | Pending |
| NAV-03 | Phase 10 | HOME | Pending |
| NAV-04 | Phase 10 | HOME | Pending |
| NAV-05 | Phase 10 | HOME | Pending |
| SETUP-01 | Phase 10 | HOME | Pending |
| SETUP-02 | Phase 10 | HOME | Pending |
| SETUP-03 | Phase 10 | HOME | Pending |
| WAY-01 | Phase 11 | HOME | Pending |
| WAY-02 | Phase 11 | HOME | Pending |
| WAY-03 | Phase 11 | HOME | Pending |
| WAY-04 | Phase 11 | HOME | Pending |
| WAY-05 | Phase 11 | HOME | Pending |
| DISP-01 | Phase 11 | HOME | Pending |
| DISP-02 | Phase 11 | HOME | Pending |
| APP-01 | Phase 12 | HOME | Pending |
| APP-02 | Phase 12 | HOME | Pending |
| APP-03 | Phase 12 | HOME | Pending |
| APP-04 | Phase 12 | HOME | Pending |
| APP-05 | Phase 12 | HOME | Pending |
| APP-06 | Phase 12 | HOME | Pending |
| APP-07 | Phase 12 | HOME | Pending |
| APP-08 | Phase 12 | HOME | Pending |
| APP-09 | Phase 12 | HOME | Pending |
| WIE-01 | Phase 12 | HOME | Pending |
| WIE-02 | Phase 12 | HOME | Pending |
| WIE-03 | Phase 14 | LAB/on-site | Pending |

**Coverage:**
- v1.0 requirements: 35 total — mapped to phases 1-8
- v2.0 requirements: 41 total — mapped to phases 9-14
- Total mapped: 76
- Unmapped: 0
- HOME phases (code only, no robot): 1-5, 9, 10, 11, 12
- LAB phases (real hardware): 7, 8, 13
- HOME+LAB phases (code at home, hardware test in lab): 6
- LAB/on-site phase: 14 (Student Union WiE deployment)

---
*Requirements defined: 2026-03-09*
*Last updated: 2026-03-13 — v2.0 traceability rewritten with HOME/LAB split (phases 9-14)*
