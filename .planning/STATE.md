---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 03-vision-pipeline-code plan 01 (03-01-PLAN.md)
last_updated: "2026-03-10T07:55:45.820Z"
last_activity: 2026-03-09 -- Roadmap restructured for HOME/LAB split
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 7
  completed_plans: 6
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** Robot reliably isolates and converses with one person in a noisy conference room using directional audio and visual cues
**Current focus:** Phase 1: Dependency Setup & Stub API Fixes

## Current Position

Phase: 1 of 8 (Dependency Setup & Stub API Fixes)
Plan: 0 of 3 in current phase
Status: Ready to plan
Last activity: 2026-03-09 -- Roadmap restructured for HOME/LAB split

Progress: [███░░░░░░░] 33%

## Work Location

**HOME phases (1-5):** Code writing, stub fixes, unit tests with mocked models -- no GPU needed
**MIXED phase (6):** Android code at home, hardware test in lab
**LAB phases (7-8):** GPU validation, E2E integration -- RTX 5070 + robot required

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01-dependency-setup-stub-api-fixes P01 | 15 | 2 tasks | 11 files |
| Phase 01-dependency-setup-stub-api-fixes P02 | 10 | 2 tasks | 4 files |
| Phase 01-dependency-setup-stub-api-fixes P03 | 4m18s | 2 tasks | 4 files |
| Phase 02-tts-pipeline-code P02 | 2m13s | 1 tasks | 2 files |
| Phase 02-tts-pipeline-code P01 | 8m | 1 tasks | 2 files |
| Phase 03-vision-pipeline-code P01 | 2min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: HOME/LAB split -- maximize home work, batch lab testing
- [Roadmap]: Kokoro TTS before Dolphin (lowest risk, pip-installable, immediately testable)
- [Roadmap]: Vision pipeline before Dolphin (Dolphin REQUIRES lip frames as input)
- [Roadmap]: AEC research in Phase 5 (replaces mic gating for barge-in support)
- [Research]: Every stub API is wrong -- all need rewriting before model work
- [Research]: Dolphin not pip-installable -- vendor source, import from look2hear.models
- [Phase 01-dependency-setup-stub-api-fixes]: Vendor look2hear/ by copy into project root (git-tracked) rather than symlink — portable across machines
- [Phase 01-dependency-setup-stub-api-fixes]: Use xfail markers for stub-correctness RED tests — Plans 02/03 will make them green
- [Phase 01-dependency-setup-stub-api-fixes]: Fix EventBus stale loop by calling asyncio.get_running_loop() per-emit instead of caching self._loop
- [Phase 01-dependency-setup-stub-api-fixes]: Dolphin takes mono [1,samples] audio — always average multi-channel input before model call
- [Phase 01-dependency-setup-stub-api-fixes]: Grayscale conversion to 88x88 happens inside _run_dolphin, not in LipExtractor — keeps pipeline RGB-native
- [Phase 01-dependency-setup-stub-api-fixes]: TTSEngine voice defaults to af_heart via getattr on TTSConfig — no schema change needed
- [Phase 01-dependency-setup-stub-api-fixes]: Log message 'LiveKit turn detector' retained without hyphen for user clarity; tests check lowercase 'livekit' which is absent
- [Phase 02-tts-pipeline-code]: Tests went straight to GREEN: production code (protocol.py, manager.py) already implemented TTS_AUDIO 0x05 wiring correctly
- [Phase 02-tts-pipeline-code]: emit_async replaces emit() throughout TTSEngine — TTS_END cannot race ahead of audio chunks
- [Phase 02-tts-pipeline-code]: hasattr(audio, 'cpu') duck-type guard avoids importing torch in tts.py
- [Phase 03-vision-pipeline-code]: VisionConfig.lip_roi_size corrected to (88,88) — Dolphin requires exactly 88x88; 96x96 would produce wrong-shaped tensors
- [Phase 03-vision-pipeline-code]: GazeEstimator L2CS tests went straight to GREEN — production code already handles empty yaw/pitch and RuntimeError

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: PyTorch nightly + sm_120 + NeMo is a fragile combination -- Phase 7 may need debugging
- [Research]: Dolphin real-time streaming untested -- model designed for 4-second file windows
- [Research]: L2CS-Net weights hosted on Google Drive may be flaky to download
- [Research]: CAE branch needs revert-the-revert (not direct merge)

## Session Continuity

Last session: 2026-03-10T07:55:45.818Z
Stopped at: Completed 03-vision-pipeline-code plan 01 (03-01-PLAN.md)
Resume file: None
