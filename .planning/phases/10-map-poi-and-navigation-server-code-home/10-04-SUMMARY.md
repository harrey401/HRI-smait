---
phase: 10-map-poi-and-navigation-server-code-home
plan: "04"
subsystem: navigation
tags: [nav-controller, poi, chassis, websocket, event-bus, tdd]

requires:
  - phase: 10-03
    provides: POIKnowledgeBase with load/resolve/fetch_markers
  - phase: 10-02
    provides: MapManager and /get_map_info chassis service
  - phase: 09
    provides: ChassisClient with call_service and send_cancel_navigation

provides:
  - NavController.navigate_to — resolves human POI names, calls /poi service, emits NAV_STARTED/NAV_FAILED
  - NavController.cancel_navigation — calls send_cancel_navigation + emits NAV_CANCELLED
  - NavController.calculate_distance — calls /calculate_distance, returns float meters
  - NavController._on_nav_status — translates chassis status codes 2/3/4 to NAV events
  - NavController.on_chassis_connected (SETUP-03) — auto-detects active floor and loads POI config
  - SETUP-01 documentation — new location setup workflow in nav_controller.py

affects:
  - phase-11-wayfinding-llm-tools
  - phase-13-lab-integration

tech-stack:
  added: []
  patterns:
    - "Sync event handler _on_nav_status translates chassis integer status codes to typed EventType enums"
    - "Async on_chassis_connected as CHASSIS_CONNECTED handler wires MapManager + POIKnowledgeBase startup"
    - "Defensive typo handling: check both avaliable_list and available_list in chassis responses"

key-files:
  created: []
  modified:
    - smait/navigation/nav_controller.py
    - tests/unit/test_nav_controller.py

key-decisions:
  - "NavController owns SETUP-03 startup wiring (not MapManager) — it depends on both MapManager and POIKnowledgeBase so is the right coordinator"
  - "on_chassis_connected uses /get_map_info with cmd=0 — consistent with MapManager list_maps pattern"
  - "navigate_to uses poi_kb.resolve() or fallback to original name — allows chassis marker names to pass through directly"
  - "NAV_FAILED available list field named 'available' in event data (not avaliable) — clean internal event despite protocol typo"

patterns-established:
  - "Chassis status codes (0=pending, 1=active, 2=preempted, 3=succeeded, 4=aborted) mapped to NAV events in _on_nav_status"
  - "SETUP-01 workflow documented at module level in nav_controller.py for new location onboarding"

requirements-completed:
  - NAV-01
  - NAV-02
  - NAV-03
  - NAV-04
  - NAV-05
  - SETUP-01
  - SETUP-03

duration: 12min
completed: 2026-03-14
---

# Phase 10 Plan 04: NavController Summary

**NavController implemented in TDD GREEN — POI name resolution + /poi service call + chassis status monitoring + SETUP-03 auto floor detection on chassis connect**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-14T09:00:00Z
- **Completed:** 2026-03-14T09:12:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- NavController fully implemented — all 5 NAV requirement tests passing
- navigate_to resolves human names via POIKnowledgeBase then calls /poi service, emits NAV_STARTED or NAV_FAILED
- _on_nav_status handler translates status codes 2/3/4 to NAV_CANCELLED/NAV_ARRIVED/NAV_FAILED events
- cancel_navigation calls chassis send_cancel_navigation + emits NAV_CANCELLED
- calculate_distance calls /calculate_distance service with all 6 args, returns float meters
- SETUP-03: on_chassis_connected auto-detects active floor via /get_map_info + loads POI config + emits MAP_ACTIVE_FLOOR
- SETUP-01: New location setup workflow documented in nav_controller.py module-level comment block
- Added test_navigate_to_poi_not_found (NAV-02) and test_startup_auto_detect (SETUP-03) — 6 total NavController tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement NavController (TDD GREEN)** - `7f4b76c` (feat)
2. **Task 2: SETUP-03 startup wiring and SETUP-01 documentation** - `f95706c` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified
- `smait/navigation/nav_controller.py` — Full NavController implementation (stub replaced): navigate_to, cancel_navigation, calculate_distance, _on_nav_status, on_chassis_connected, navigating property
- `tests/unit/test_nav_controller.py` — Added test_navigate_to_poi_not_found (NAV-02) and test_startup_auto_detect (SETUP-03); all 6 tests pass

## Decisions Made
- NavController owns SETUP-03 startup wiring because it is the coordinator between MapManager results and POIKnowledgeBase loads — placing this in MapManager would create a dependency inversion
- on_chassis_connected uses /get_map_info with cmd=0, matching the established MapManager pattern
- navigate_to falls back to original poi_name when resolve() returns None — allows chassis marker names to bypass the KB lookup
- NAV_FAILED event data uses key "available" (not "avaliable") — clean internal event naming despite the chassis protocol typo being preserved in call sites

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing test failures in test_audio_pipeline, test_barge_in, test_dolphin_separator, and test_engagement — all confirmed pre-existing before our changes via git stash verification. Out of scope per deviation rules. Logged to deferred-items.

## Next Phase Readiness
- All Phase 10 spatial data layer complete: MapManager + POIKnowledgeBase + NavController
- Phase 11 (Wayfinding LLM Tools) can depend on NavController.navigate_to, calculate_distance, and MAP_ACTIVE_FLOOR event
- NavController.on_chassis_connected wired — Phase 13 lab integration just needs real chassis WS connection

---
*Phase: 10-map-poi-and-navigation-server-code-home*
*Completed: 2026-03-14*
