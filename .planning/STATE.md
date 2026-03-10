---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 05-turn-taking-aec-code plan 01 (05-01-PLAN.md)
last_updated: "2026-03-10T10:05:57.739Z"
last_activity: 2026-03-09 -- Roadmap restructured for HOME/LAB split
progress:
  total_phases: 8
  completed_phases: 4
  total_plans: 11
  completed_plans: 10
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
| Phase 03-vision-pipeline-code P02 | 4min | 2 tasks | 2 files |
| Phase 04-speaker-separation-code P01 | 2m42s | 2 tasks | 3 files |
| Phase 04-speaker-separation-code P02 | 3m47s | 2 tasks | 3 files |
| Phase 05-turn-taking-aec-code P01 | 4min | 2 tasks | 6 files |

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
- [Phase 03-vision-pipeline-code]: Walking-past test must prime area history with gaze=OFF first: filter requires >=3 entries before suppressing approach, gaze=ON at t=0 causes APPROACHING before history exists
- [Phase 03-vision-pipeline-code]: Patch smait.perception.face_tracker.mp (module-level alias) not mediapipe.solutions.face_mesh — mediapipe.solutions is not installed in this environment
- [Phase 04-speaker-separation-code]: Early exit passthrough before _run_dolphin when lip_frames=[] prevents audio-only Dolphin TypeError crash
- [Phase 04-speaker-separation-code]: torch.inference_mode() replaces torch.no_grad() in DolphinSeparator — faster, disables grad tracking entirely
- [Phase 04-speaker-separation-code]: main.py always passes segment.cae_audio to Dolphin with channels=1 — raw 4-channel is wrong input for Dolphin mono model
- [Phase 04-speaker-separation-code]: Test face areas equalized (10000 each) so DOA angular proximity is the discriminating factor — original plan face areas (15000 vs 8000) were mathematically insufficient for the formula to produce the expected winner
- [Phase 04-speaker-separation-code]: DOA scoring: max(0.5, 1.0 - angular_distance/90.0) with frame_width=640, camera_fov_deg=60; flat 1.2x bonus removed
- [Phase 05-turn-taking-aec-code]: Hallucination filter runs before short confidence check — recognised phrases must always carry hallucination_phrase label
- [Phase 05-turn-taking-aec-code]: _extract_confidence handles both list[Hypothesis] and single Hypothesis objects from NeMo transcribe()
- [Phase 05-turn-taking-aec-code]: hard_cutoff_ms default changed from 1500 to 1800 to match vad_silence_ms threshold

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: PyTorch nightly + sm_120 + NeMo is a fragile combination -- Phase 7 may need debugging
- [Research]: Dolphin real-time streaming untested -- model designed for 4-second file windows
- [Research]: L2CS-Net weights hosted on Google Drive may be flaky to download
- [Research]: CAE branch needs revert-the-revert (not direct merge)

## Session Continuity

Last session: 2026-03-10T10:05:57.737Z
Stopped at: Completed 05-turn-taking-aec-code plan 01 (05-01-PLAN.md)
Resume file: None
