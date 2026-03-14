---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: phases
status: planning
stopped_at: Phase 12 context gathered
last_updated: "2026-03-14T20:44:48.309Z"
last_activity: 2026-03-13 — Roadmap rewritten with HOME/LAB split (phases 9-14)
progress:
  total_phases: 14
  completed_phases: 8
  total_plans: 25
  completed_plans: 23
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
- [Phase 09]: Dual fragment handling: rosbridge op=fragment AND raw partial-JSON buffer for two-frame split messages
- [Phase 09]: event_bus property exposed publicly on ChassisClient (client.event_bus) as tests access it directly
- [Phase 10]: NavigationConfig stored as separate dataclass from ChassisConfig for clean separation
- [Phase 10]: /global_path auto-subscribed in _setup_subscriptions; subscribe_topic is manual API for MapManager
- [Phase 10]: Flat response shape {'waypoints': [...]} takes priority over nested in fetch_markers for protocol compatibility
- [Phase 10]: isinstance(result, dict) guard in fetch_markers for AsyncMock safety and real-world error resilience
- [Phase 10]: decode_map_png dual-format: flat dict for tests, nested rosbridge msg for real chassis
- [Phase 10]: render_map returns placeholder grey PNG when no map loaded — prevents callers from getting None
- [Phase 10]: switch_map calls /layered_map_cmd with cmd=7 (matches test assertion and real protocol)
- [Phase 10]: NavController owns SETUP-03 startup wiring — coordinator between /get_map_info results and POIKnowledgeBase loads
- [Phase 10]: navigate_to falls back to original poi_name when resolve() returns None — chassis marker names bypass KB lookup
- [Phase Phase 10]: switch_map calls /node_manager_control with cmd=7 (not /layered_map_cmd which is for listing)
- [Phase 11]: _on_display_map guards empty png_bytes; _on_display_nav_status guards isinstance(data, dict); DISPLAY_MAP/NAV_STATUS added idempotently (Plan 01 had not run yet)
- [Phase 11]: WayfindingManager owns DISPLAY_MAP and DISPLAY_NAV_STATUS emission (not MapManager or NavController)
- [Phase 11]: MapManager._poi_positions cache populated via POI_LIST_UPDATED subscription in start() — avoids tight coupling with POIKnowledgeBase internals
- [Phase 11]: Ollama path does NOT receive tools= parameter — local LLMs have unreliable tool-calling
- [Phase 11]: WayfindingManager subscribes to NAV_ARRIVED/NAV_FAILED in __init__ for verbal arrival/failure confirmations
- [Phase 11]: DIALOGUE_RESPONSE emitted by WayfindingManager with model_used=wayfinding to distinguish from LLM responses

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

Last session: 2026-03-14T20:44:48.307Z
Stopped at: Phase 12 context gathered
Resume file: .planning/phases/12-android-app-rebuild-and-wie-theme-home/12-CONTEXT.md
Next step: `/gsd:plan-phase 9`
