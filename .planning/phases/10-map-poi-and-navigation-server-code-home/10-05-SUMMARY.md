---
phase: 10-map-poi-and-navigation-server-code-home
plan: 05
subsystem: navigation
tags: [map-manager, event-bus, rosbridge, png-decode, test-coverage]

# Dependency graph
requires:
  - phase: 10-map-poi-and-navigation-server-code-home
    provides: MapManager implementation from plans 10-02/10-04

provides:
  - Corrected switch_map() using /node_manager_control (MAP-03 unblocked)
  - Event handler tests for _on_map_update, _on_pose_update, _on_path_update
  - Rosbridge nested format test for decode_map_png
  - MapManager coverage raised from 63% to 90%

affects: [phase-13-lab-integration, phase-10-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sync event handler subscription in tests: directly subscribe handler to bus without calling start()"
    - "Flat-format vs nested rosbridge format: both handled in decode_map_png dual-branch"

key-files:
  created: []
  modified:
    - smait/navigation/map_manager.py
    - tests/unit/test_map_manager.py

key-decisions:
  - "switch_map calls /node_manager_control with cmd=7 (not /layered_map_cmd which is for listing)"
  - "Event handler tests subscribe handlers directly (not via start()) to avoid async subscribe_topic dependency"

patterns-established:
  - "Test event handlers by subscribing them directly to the bus — bypass start() lifecycle"
  - "MAP_RENDERED emission verified by subscribing a capture callback before emitting the trigger event"

requirements-completed: [MAP-01, MAP-03, MAP-04]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 10 Plan 05: Gap Closure — switch_map Protocol Fix + Event Handler Tests Summary

**switch_map() corrected to /node_manager_control endpoint and event handler paths fully tested, raising MapManager coverage from 63% to 90%**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-14T09:26:19Z
- **Completed:** 2026-03-14T09:28:18Z
- **Tasks:** 2 of 2
- **Files modified:** 2

## Accomplishments

- Fixed MAP-03 blocker: switch_map() was calling `/layered_map_cmd` (the list endpoint); corrected to `/node_manager_control` with cmd=7 per chassis protocol spec
- Added four new tests covering the three untested event handler paths: `_on_map_update` (flat format), `_on_map_update` (rosbridge nested format), `_on_pose_update` (with/without map), `_on_path_update`
- Coverage rose from 63% to 90%, exceeding the 80% target

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix switch_map protocol and update test assertion** - `64990d2` (fix)
2. **Task 2: Add event handler and rosbridge format tests** - `038f5fb` (test)

## Files Created/Modified

- `smait/navigation/map_manager.py` - Fixed switch_map() endpoint from /layered_map_cmd to /node_manager_control
- `tests/unit/test_map_manager.py` - Updated assertion in test_switch_map; added 4 new event handler tests

## Decisions Made

- Event handler tests subscribe handlers directly to the bus (not via `start()`) so they can be synchronous tests without needing async `subscribe_topic` calls on the mock chassis. This keeps the tests simple and focused on handler behavior.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The first run of new tests failed because `_on_map_update`, `_on_pose_update`, and `_on_path_update` were not registered on the bus (registration happens in `start()`). Fixed by explicitly subscribing each handler to the bus before emitting the trigger event. This is the correct pattern for testing synchronous event handlers without calling the full lifecycle `start()` method.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- MAP-01, MAP-03, MAP-04 are now fully satisfied
- Phase 10 gap closure is complete — all MAP requirements met
- Phase 13 lab integration can proceed with correct switch_map endpoint

---
*Phase: 10-map-poi-and-navigation-server-code-home*
*Completed: 2026-03-14*
