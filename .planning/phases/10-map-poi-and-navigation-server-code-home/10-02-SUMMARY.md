---
phase: 10-map-poi-and-navigation-server-code-home
plan: "02"
subsystem: navigation
tags: [navigation, map, PIL, base64, world-to-pixel, chassis-client, tdd-green]

requires:
  - phase: 10-01
    provides: MapManager skeleton with NotImplementedError stubs, EventBus with MAP_* events, ChassisClient with subscribe_topic and call_service

provides:
  - MapManager fully implemented with map decode, render, list, switch
  - world_to_pixel transform with Y-axis flip and bounds clamping
  - draw_robot_arrow with body circle and directional line
  - decode_map_png supporting both flat and nested rosbridge message formats
  - Placeholder PNG render when no map received (prevents test/display errors)
  - Auto-detect on CHASSIS_CONNECTED via concurrent asyncio.gather

affects: [10-03, 10-04, phase-11-wayfinding-display]

tech-stack:
  added: [PIL ImageDraw, io.BytesIO for PNG encode/decode, math.cos/sin for arrow rendering, asyncio.gather for concurrent service calls]
  patterns: [TDD GREEN phase, dual-format message parsing, placeholder-first rendering, async event handler pattern]

key-files:
  created: []
  modified:
    - smait/navigation/map_manager.py

key-decisions:
  - "decode_map_png supports two formats: flat dict (tests/mock) and nested rosbridge msg structure (real chassis)"
  - "render_map returns placeholder grey PNG when no map loaded — prevents callers from getting None/error"
  - "switch_map calls /layered_map_cmd with cmd=7 (matches test assertion; real chassis protocol)"
  - "_on_chassis_connected uses asyncio.gather for concurrent /get_map_info + /layered_map_cmd calls"
  - "MAP_ACTIVE_FLOOR extraction handles both flat building_name key and nested list_info format"

patterns-established:
  - "dual-format parse: check for 'msg' key (nested rosbridge) else fall back to flat format"
  - "placeholder render: return valid image bytes even with no data loaded"

requirements-completed: [MAP-01, MAP-02, MAP-03, MAP-04]

duration: 15min
completed: "2026-03-14"
---

# Phase 10 Plan 02: MapManager GREEN Phase Summary

**MapManager fully implemented — base64 PNG decode with Y-axis flip world-to-pixel transform, robot arrow rendering, path polyline overlay, map list/switch via chassis service calls, and CHASSIS_CONNECTED auto-detect; all 7 tests GREEN.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-14T08:00:00Z
- **Completed:** 2026-03-14T08:15:00Z
- **Tasks:** 1 (single GREEN implementation task)
- **Files modified:** 1

## Accomplishments

- Implemented `world_to_pixel` with ROS OccupancyGrid Y-axis convention (row = height - wy_offset) and bounds clamping
- Implemented `draw_robot_arrow` with filled circle body and line direction indicator using math.cos/sin
- Implemented `decode_map_png` handling both flat dict (tests) and nested rosbridge `msg.msg.info` format
- `render_map()` returns a valid placeholder grey PNG when no map has been received yet (enables safe calls at startup)
- `MapManager.start()` wires four event subscriptions and sends map subscribe_topic request
- `_on_chassis_connected` uses `asyncio.gather` for concurrent `/get_map_info` + `/layered_map_cmd` service calls
- MAP_ACTIVE_FLOOR emission extracts building/floor from either flat or list_info response formats
- All 7 `test_map_manager.py` tests pass; 14 `test_chassis_client.py` tests still passing (no regressions)

## Task Commits

1. **MapManager GREEN implementation** - `c0f173b` (feat)

## Files Created/Modified

- `smait/navigation/map_manager.py` — Full implementation replacing all NotImplementedError stubs (285 lines added, 36 removed)

## Decisions Made

- `decode_map_png` dual-format: The test uses a flat dict (data/width/height/origin_x directly) while the real chassis sends a nested `msg.msg` structure per the rosbridge protocol. Implemented a check: if `msg.get("msg")` is not None, use nested path; otherwise flat path.
- `render_map` placeholder: Rather than raising when no map loaded, return a 100x100 grey RGBA PNG with unit-scale meta. This prevents callers from handling None and matches the test expectation.
- `switch_map` uses `/layered_map_cmd` with `cmd=7`: The test assertion checks `"/layered_map_cmd" in str(call_args)` with `7` present. Used this endpoint (matches real protocol: `/layered_map_cmd` handles both list and switch with different cmd values).
- `_on_chassis_connected` wraps gather in try/except: Timeout resilience — if either service call fails, the other result is still processed (return_exceptions=True in gather).

## Deviations from Plan

None — plan executed exactly as written. The dual-format decode was an interpretation of the existing test structure, consistent with the plan's message format description.

## Issues Encountered

None — tests passed on first run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- MapManager is fully implemented and tested GREEN
- Ready for Plan 10-03 (POIKnowledgeBase GREEN phase)
- Plan 10-04 (NavController GREEN phase) also ready
- Phase 11 display layer can use MapManager.render_map() and MAP_RENDERED events

## Self-Check: PASSED

Files verified:
- smait/navigation/map_manager.py — FOUND
- .planning/phases/10-map-poi-and-navigation-server-code-home/10-02-SUMMARY.md — FOUND

Commits verified:
- c0f173b (feat(10-02): implement MapManager) — FOUND

---
*Phase: 10-map-poi-and-navigation-server-code-home*
*Completed: 2026-03-14*
