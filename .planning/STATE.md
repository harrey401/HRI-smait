# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** Robot reliably isolates and converses with one person in a noisy conference room using directional audio and visual cues
**Current focus:** Phase 1: Environment & API Foundation

## Current Position

Phase: 1 of 7 (Environment & API Foundation)
Plan: 0 of 3 in current phase
Status: Ready to plan
Last activity: 2026-03-09 -- Roadmap created

Progress: [░░░░░░░░░░] 0%

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

- [Roadmap]: Kokoro TTS before Dolphin (lowest risk, pip-installable, immediately testable)
- [Roadmap]: Vision pipeline before Dolphin (Dolphin REQUIRES lip frames as input)
- [Roadmap]: AEC research in Phase 6 (replaces mic gating for barge-in support)

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: PyTorch nightly + sm_120 + NeMo is a fragile combination -- Phase 1 may need debugging
- [Research]: Dolphin real-time streaming untested -- model designed for 4-second file windows
- [Research]: L2CS-Net weights hosted on Google Drive may be flaky to download

## Session Continuity

Last session: 2026-03-09
Stopped at: Roadmap created, ready to plan Phase 1
Resume file: None
