---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: phases
status: planning
stopped_at: Completed 09-01-PLAN.md (chassis contracts — RED phase)
last_updated: "2026-03-14T06:53:58.560Z"
last_activity: 2026-03-13 — Roadmap rewritten with HOME/LAB split (phases 9-14)
progress:
  total_phases: 14
  completed_phases: 5
  total_plans: 17
  completed_plans: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Robot reliably isolates and converses with one person in a noisy conference room using directional audio and visual cues
**Current milestone:** v2.0 Navigation & Wayfinding
**Current focus:** Phase 9 — Chassis WebSocket Client (HOME)

## Current Position

Phase: 9 (not started)
Plan: —
Status: Roadmap complete, ready to plan Phase 9
Last activity: 2026-03-13 — Roadmap rewritten with HOME/LAB split (phases 9-14)

## v2.0 Phase Summary

| Phase | Name | Location | Requirements | Status |
|-------|------|----------|--------------|--------|
| 9 | Chassis WebSocket Client | HOME | CHAS-01 to CHAS-06 | Not started |
| 10 | Map, POI, and Navigation Server Code | HOME | MAP-01 to MAP-04, POI-01 to POI-04, NAV-01 to NAV-05, SETUP-01 to SETUP-03 | Not started |
| 11 | Wayfinding LLM Tools and Display Rendering | HOME | WAY-01 to WAY-05, DISP-01, DISP-02 | Not started |
| 12 | Android App Rebuild and WiE Theme | HOME | APP-01 to APP-09, WIE-01, WIE-02 | Not started |
| 13 | Lab Integration and Robot Verification | LAB | (integration of phases 9-11 on real chassis) | Not started |
| 14 | WiE On-Site Deployment | LAB/on-site | WIE-03 | Not started |

**Hard deadline:** WiE 2026 event — March 21, 2026

## Accumulated Context

### Decisions

- [v1.0]: HOME/LAB split — maximize home work, batch lab testing
- [v1.0]: All stub APIs rewritten before model work
- [v2.0]: Chassis communicates via WebSocket at 192.168.20.22 (JSON protocol documented in PDF)
- [v2.0]: Engineering lab already mapped via Deployment Tool
- [v2.0]: HOME/LAB split enforced — phases 9-12 are all HOME code with mock chassis; phases 13-14 are LAB only
- [v2.0]: Phase 9 defines ChassisClient interface; phases 10-12 depend on that interface (not on real hardware)
- [v2.0]: Phase 10 bundles map + POI + nav + setup — all server-side spatial data layer
- [v2.0]: Phase 11 adds LLM tool-use layer on top of Phase 10 primitives
- [v2.0]: Phase 12 is full Android app rewrite (Jetpack Compose + MVVM) plus WiE-01/02 config assets
- [v2.0]: Phase 13 is minimal lab integration — connect real chassis, verify phases 9-11 work end-to-end
- [v2.0]: Phase 14 is WiE-03 only (Student Union mapping + POI labeling) — on-site walkthrough
- [v2.0]: WIE-01 and WIE-02 moved to Phase 12 (HOME) — pure config/assets, no robot needed
- [v2.0]: WIE-03 stays in Phase 14 (on-site) — requires physical presence at Student Union
- [Phase 09]: Deferred ChassisClient import in tests (try/except) to allow pytest collection in RED state
- [Phase 09]: Used websockets 16.0 asyncio serve API with port=0 for random OS-assigned test ports

### Pending Todos

- Plan Phase 9 (chassis WebSocket client, HOME)

### Blockers/Concerns

- [v2.0]: Chassis WS connection from SMAIT server untested — may need network routing on lab WiFi
- [v2.0]: Chassis IP 192.168.20.22 may be internal network only (Android↔chassis only); Phase 13 will reveal this
- [v2.0]: Unknown if chassis WS accepts multiple simultaneous clients
- [v2.0]: Student Union not yet mapped — must be done on-site in Phase 14
- [v1.0]: Phases 7-8 still pending (camera issue + GPU validation) — will finish in lab
- [v2.0]: Starting v2 HOME phases now while v1 lab work is pending

## Session Continuity

Last session: 2026-03-14T06:53:58.557Z
Stopped at: Completed 09-01-PLAN.md (chassis contracts — RED phase)
Resume file: None
Next step: `/gsd:plan-phase 9`
