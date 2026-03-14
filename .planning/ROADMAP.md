# Roadmap: SMAIT v3 Full Hardware Integration

## Overview

Transform the SMAIT v3 HRI system from stub-based architecture to fully operational audio-visual ML pipeline. The roadmap is split into HOME phases (code writing, stub fixes, unit tests with mocked models) and LAB phases (GPU validation, hardware testing, E2E integration). This lets you maximize progress from home, then validate everything in focused lab sessions.

Milestone v2.0 (Navigation & Wayfinding) extends the roadmap from phase 9. It connects the SMAIT brain to Jackie's chassis SLAM/navigation system, enabling voice-driven wayfinding and physical guidance. The WiE event on March 21, 2026 is the hard target.

**Design principle for v2.0:** Write everything possible at home with mock chassis servers and test data, then go to the lab and do MINIMAL work to get it running on the real robot. Same pattern as v1.0 (phases 1-5 HOME, phases 7-8 LAB).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

### v1.0 phases (COMPLETE):
- [x] **Phase 1: Dependency Setup & Stub API Fixes** - Vendor Dolphin, fix all stub APIs, install packages (completed 2026-03-10)
- [x] **Phase 2: TTS Pipeline Code** - Rewrite Kokoro integration with correct KPipeline API (completed 2026-03-10)
- [x] **Phase 3: Vision Pipeline Code** - Rewrite L2CS-Net gaze, lip extraction for Dolphin format (completed 2026-03-10)
- [x] **Phase 4: Speaker Separation Code** - Rewrite Dolphin separator with correct API and tensor shapes (completed 2026-03-10)
- [x] **Phase 5: Turn-Taking & AEC Code** - VAD-based EOU, AEC research, barge-in logic (completed 2026-03-10)
- [x] **Phase 6: Android Audio Pipeline** - Merge CAE beamforming, 3 streams + DOA, AudioTrack playback (completed 2026-03-10)

### v1.0 phases (LAB — pending):
- [ ] **Phase 7: GPU Validation & Model Loading** - Verify all models on sm_120, VRAM budget, Parakeet ASR
- [ ] **Phase 8: Full Integration & Quality** - End-to-end conversation loop, latency tuning, 80% coverage

### v2.0 phases — Navigation & Wayfinding (HOME phases first):
- [x] **Phase 9: Chassis WebSocket Client (HOME)** - Full client with mock chassis server, pose/nav/state subscriptions, auto-reconnect, 100% unit tested (completed 2026-03-14)
- [x] **Phase 10: Map, POI, and Navigation Server Code (HOME)** - Map retrieval and rendering, POI knowledge base, nav commands — all mocked for home testing (completed 2026-03-14)
- [ ] **Phase 11: Wayfinding LLM Tools and Display Rendering (HOME)** - LLM function-calling tools, PIL map rendering with overlays, display dispatch
- [ ] **Phase 12: Android App Rebuild and WiE Theme (HOME)** - Complete Jetpack Compose rewrite, all screens, JSON theme system, WiE branding and event config
- [ ] **Phase 13: Lab Integration and Robot Verification (LAB)** - Connect phases 9-11 to real chassis, verify map/nav/wayfinding end-to-end on Jackie
- [ ] **Phase 14: WiE On-Site Deployment (LAB/on-site)** - Deploy app to Jackie touchscreen, map Student Union, label WiE POIs, run full demo

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
- [x] 01-01-PLAN.md — Vendor Dolphin, install dependencies, bootstrap test infrastructure, fix EventBus
- [x] 01-02-PLAN.md — Fix DolphinSeparator and TTSEngine stub imports and API usage
- [x] 01-03-PLAN.md — Fix GazeEstimator arch param and strip LiveKit from EOUDetector

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
- [x] 02-01-PLAN.md — Fix TTSEngine emit_async streaming, GPU-safe audio conversion, streaming tests
- [x] 02-02-PLAN.md — Protocol and ConnectionManager TTS audio forwarding tests

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
- [x] 03-01-PLAN.md — Fix lip_roi_size config to 88x88, write LipExtractor and GazeEstimator L2CS tests
- [x] 03-02-PLAN.md — Write EngagementDetector state machine and FaceTracker IOU tests

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

**Plans:** 2/2 plans complete

Plans:
- [x] 04-01-PLAN.md — Fix DolphinSeparator passthrough on empty lip_frames, audio routing bug in main.py, inference_mode
- [x] 04-02-PLAN.md — AudioPipeline VAD/ring buffer tests, DOA-to-face angular proximity scoring

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

**Plans:** 2/2 plans complete

Plans:
- [x] 05-01-PLAN.md — Rewrite EOUDetector with VAD-prob silence tracking, hallucination filter tests + NeMo confidence
- [x] 05-02-PLAN.md — BARGE_IN event + barge-in VAD path in AudioPipeline, cancellable TTS, SoftwareAEC class

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

**Plans:** 3 plans (2 complete, 1 gap closure pending)

Plans:
- [x] 06-01-PLAN.md — CaeAudioManager with fixed channel adapter, dual stream (0x01+0x03), JSON DOA, MainActivity wiring
- [x] 06-02-PLAN.md — TtsAudioPlayer AudioTrack 24kHz PCM16 playback, binary WebSocket handler for 0x05 frames
- [ ] 06-03-PLAN.md — Gap closure: run Gradle build and unit tests in lab (JDK required)

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

---

## v2.0 Phase Details — Navigation & Wayfinding

### Phase 9: Chassis WebSocket Client (HOME)
**Goal**: The chassis WebSocket client is fully implemented, unit-tested against a mock server, and ready to connect to real hardware
**Location**: HOME
**Depends on**: Phase 8 (v1.0 complete; v2.0 chassis client can be written independently)
**Requirements**: CHAS-01, CHAS-02, CHAS-03, CHAS-04, CHAS-05, CHAS-06
**Success Criteria** (what must be TRUE):
  1. A mock chassis WebSocket server (in-process test fixture) accepts connections and responds to all chassis protocol messages so tests run without real hardware
  2. ChassisClient connects to a configurable IP/port, sends the handshake, and receives pose updates — verified against the mock server
  3. Pose subscription delivers (x, y, theta) updates at the configured interval and fires a SMAIT internal event per update
  4. Navigation status events (running/success/failed/cancelled) arrive from the mock chassis and are translated into SMAIT EventBus events
  5. Global robot state (battery, velocity, control_state) is retrievable via a single call to the mock chassis
  6. Soft e-stop sends the correct chassis protocol message and the client retries with exponential backoff on mock disconnect

**Plans:** 2/2 plans complete

Plans:
- [x] 09-01-PLAN.md — Contracts: ChassisConfig, EventType members, MockChassisServer fixture, failing test suite
- [x] 09-02-PLAN.md — Implementation: ChassisClient with connection, subscriptions, message routing, soft e-stop, reconnect

### Phase 10: Map, POI, and Navigation Server Code (HOME)
**Goal**: All server-side spatial data code is implemented and unit-tested with sample map data — no real chassis required
**Location**: HOME
**Depends on**: Phase 9 (ChassisClient interface defined; map/nav calls use same client)
**Requirements**: MAP-01, MAP-02, MAP-03, MAP-04, POI-01, POI-02, POI-03, POI-04, NAV-01, NAV-02, NAV-03, NAV-04, NAV-05, SETUP-01, SETUP-02, SETUP-03
**Success Criteria** (what must be TRUE):
  1. MapManager retrieves a LIDAR occupancy grid PNG from the mock chassis and renders a new PNG with the robot position overlaid as a directional arrow — verified with a sample map image
  2. MapManager lists available maps/buildings/floors and can switch the active map by sending the correct chassis protocol message
  3. POI knowledge base loads human-friendly name mappings from a per-building JSON config file (e.g., "room ENG192" maps to chassis marker "eng192") and supports add/remove marker operations
  4. NavController sends navigate-to-POI, cancel-navigation, and calculate-distance commands to the mock chassis using the correct protocol messages
  5. Path subscription delivers waypoints from the mock chassis and the rendered map image shows the planned path as an overlay
  6. On startup, the server reads available maps from the chassis, selects the active floor, and loads the matching POI JSON config file automatically

**Plans:** 4/4 plans complete

Plans:
- [ ] 10-01-PLAN.md — Core infrastructure: EventTypes, NavigationConfig, ChassisClient extensions, module skeletons, RED test suites
- [ ] 10-02-PLAN.md — MapManager TDD: map PNG decode, world-to-pixel transform, rendering, list/switch maps
- [ ] 10-03-PLAN.md — POIKnowledgeBase TDD: marker CRUD, human-name resolution, JSON config loading
- [ ] 10-04-PLAN.md — NavController TDD: nav commands, status monitoring, distance calc, startup wiring, SETUP docs

### Phase 11: Wayfinding LLM Tools and Display Rendering (HOME)
**Goal**: Users can ask Jackie "where is X?" or "take me to X" in code — LLM tools, verbal responses, and map rendering all work with mock data
**Location**: HOME
**Depends on**: Phase 10 (MapManager, POI knowledge base, NavController interfaces defined)
**Requirements**: WAY-01, WAY-02, WAY-03, WAY-04, WAY-05, DISP-01, DISP-02
**Success Criteria** (what must be TRUE):
  1. LLM dialogue manager has a registered `query_location` tool that accepts a natural-language location name, resolves it via the POI knowledge base, and returns the matched POI name and map coordinates
  2. LLM dialogue manager has a registered `navigate_to` tool that accepts a resolved POI name, issues the nav command via NavController, and returns a verbal confirmation string
  3. When the `query_location` tool fires, the rendered map image (with highlighted destination) is dispatched to the display channel and verified to contain the correct visual overlay using a test map fixture
  4. When `navigate_to` fires, the robot emits a verbal status update (e.g., "On my way to ENG192") and the display updates to show navigating status — both verified with mock chassis and mock LLM
  5. When the mock chassis emits navigation_success or navigation_failed, the verbal response and display update are triggered correctly

**Plans**: TBD

### Phase 12: Android App Rebuild and WiE Theme (HOME)
**Goal**: Jackie's touchscreen app is fully rebuilt with Jetpack Compose, ships WiE 2026 branding, and all WebSocket streams are preserved
**Location**: HOME
**Depends on**: Phase 11 (server sends map images and nav status messages — interface stable)
**Requirements**: APP-01, APP-02, APP-03, APP-04, APP-05, APP-06, APP-07, APP-08, APP-09, WIE-01, WIE-02
**Success Criteria** (what must be TRUE):
  1. App builds successfully with Jetpack Compose + MVVM; all existing WebSocket streams (audio 0x01/0x03, video 0x02, TTS 0x05) are received and handled identically to the old app
  2. Theming system reads a JSON config file and applies colors, fonts, event name, and logo with no code changes required — verified by swapping WiE theme vs a default theme at runtime
  3. Home screen renders event-specific cards (Guided Tour, Ask Me Anything, Facilities, etc.) driven by the JSON config, using WiE 2026 branding by default
  4. Navigation screen renders the map image received from the server, with robot position, path, and destination POI label visible
  5. Conversation screen shows live transcript text updating in real time, a mic activity indicator, and a robot avatar
  6. Facilities/wayfinding screen lists POIs from the server's knowledge base and a "Take me there" tap triggers the navigate_to flow
  7. Event info screen shows schedule, speakers, and venue map loaded from the JSON config — WiE 2026 content populated

**Plans**: TBD

### Phase 13: Lab Integration and Robot Verification (LAB)
**Goal**: Phases 9-12 are connected to the real chassis and Jackie's touchscreen — the wayfinding system works on actual hardware
**Location**: LAB
**Depends on**: Phases 9, 10, 11, 12 (all HOME code complete and unit-tested)
**Requirements**: (integration verification of all v2.0 server requirements against real chassis)
**Success Criteria** (what must be TRUE):
  1. ChassisClient connects to chassis at 192.168.20.22, receives real pose updates, and the map renders Jackie's actual position on the Engineering lab floor plan
  2. Real LIDAR occupancy grid PNG is retrieved from the chassis and displayed on Jackie's touchscreen with robot position overlaid correctly
  3. Navigate-to-POI command causes Jackie to physically move toward the target location and the app navigation screen tracks progress in real time
  4. Voice command "where is ENG192?" causes Jackie to show the map with a highlighted path and give a verbal response — end-to-end on the real robot
  5. Voice command "take me to the bathroom" causes Jackie to navigate to the correct POI and verbally confirm arrival

**Plans**: TBD

### Phase 14: WiE On-Site Deployment (LAB/on-site)
**Goal**: Jackie is fully configured for the WiE 2026 event at the Student Union — POIs labeled, app deployed, demo verified
**Location**: LAB/on-site (Student Union)
**Depends on**: Phase 13 (robot integration verified), Phase 12 (app deployed to Jackie)
**Requirements**: WIE-03
**Success Criteria** (what must be TRUE):
  1. Student Union is mapped via Deployment Tool (or confirmed already mapped) and the map loads in SMAIT correctly
  2. WiE POIs are labeled in the location knowledge base: registration desk, keynote hall, panel rooms, networking area — all reachable by voice command on the actual Student Union floor
  3. A full end-to-end demo runs at the Student Union: attendee asks "where is the keynote hall?" → Jackie shows map + verbal directions → "take me to the keynote hall" → Jackie navigates there

**Plans**: TBD

---

## Progress

**v1.0 Execution Order:**
- HOME: Phases 1 -> 2, 3 (parallel) -> 4 -> 5 -> 6 (code only)
- LAB: Phase 6 (hardware test) -> 7 -> 8

**v2.0 Execution Order:**
- HOME: Phase 9 -> Phase 10 -> Phase 11 -> Phase 12 (all parallelizable after Phase 9 interface is defined)
- LAB: Phase 13 (chassis integration) -> Phase 14 (WiE on-site)
- WiE hard deadline: March 21, 2026

| Phase | Location | Plans Complete | Status | Completed |
|-------|----------|----------------|--------|-----------|
| 1. Dependency Setup & Stub API Fixes | HOME | 3/3 | Complete | 2026-03-10 |
| 2. TTS Pipeline Code | HOME | 2/2 | Complete | 2026-03-10 |
| 3. Vision Pipeline Code | HOME | 2/2 | Complete | 2026-03-10 |
| 4. Speaker Separation Code | HOME | 2/2 | Complete | 2026-03-10 |
| 5. Turn-Taking & AEC Code | HOME | 2/2 | Complete | 2026-03-10 |
| 6. Android Audio Pipeline | HOME+LAB | 2/3 | Gap closure pending | 2026-03-10 |
| 7. GPU Validation & Model Loading | LAB | 0/2 | Not started | - |
| 8. Full Integration & Quality | LAB | 0/3 | Not started | - |
| 9. Chassis WebSocket Client (HOME) | HOME | 2/2 | Complete | 2026-03-14 |
| 10. Map, POI, and Navigation Server Code (HOME) | 4/4 | Complete   | 2026-03-14 | - |
| 11. Wayfinding LLM Tools and Display Rendering (HOME) | HOME | 0/? | Not started | - |
| 12. Android App Rebuild and WiE Theme (HOME) | HOME | 0/? | Not started | - |
| 13. Lab Integration and Robot Verification (LAB) | LAB | 0/? | Not started | - |
| 14. WiE On-Site Deployment (LAB/on-site) | LAB/on-site | 0/? | Not started | - |
