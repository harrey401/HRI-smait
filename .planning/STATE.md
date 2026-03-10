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

Progress: [░░░░░░░░░░] 0%

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: PyTorch nightly + sm_120 + NeMo is a fragile combination -- Phase 7 may need debugging
- [Research]: Dolphin real-time streaming untested -- model designed for 4-second file windows
- [Research]: L2CS-Net weights hosted on Google Drive may be flaky to download
- [Research]: CAE branch needs revert-the-revert (not direct merge)

## Session Continuity

Last session: 2026-03-09
Stopped at: Roadmap restructured for HOME/LAB, ready to plan Phase 1
Resume file: None
