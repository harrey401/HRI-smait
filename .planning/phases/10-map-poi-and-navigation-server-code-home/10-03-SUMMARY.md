---
phase: 10-map-poi-and-navigation-server-code-home
plan: "03"
subsystem: navigation
tags: [poi, chassis, json-config, name-resolution, crud, asyncio]

# Dependency graph
requires:
  - phase: 10-01
    provides: NavigationConfig dataclass, POI_LIST_UPDATED and POI_CONFIG_MISSING events, POIKnowledgeBase skeleton
  - phase: 09-02
    provides: ChassisClient with call_service and send_insert_marker methods

provides:
  - POIKnowledgeBase fully implemented with marker CRUD and case-insensitive name resolution
  - load() reads per-floor JSON config from data/poi/{building}/{floor}.json
  - resolve() performs case-insensitive lookup with chassis marker fallback
  - fetch_markers() calls /marker_operation/get_markers and emits POI_LIST_UPDATED
  - add_marker() and delete_marker() complete the CRUD set via chassis protocol

affects:
  - 10-04 (NavController depends on POIKnowledgeBase.resolve for destination lookup)
  - 11 (LLM wayfinding tools use POIKnowledgeBase for location resolution)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Defensive dict parsing for protocol typos (avaliable_list vs available_list)
    - Lowercase key normalization for case-insensitive config lookups
    - isinstance(result, dict) guard for async mock compatibility in tests

key-files:
  created: []
  modified:
    - smait/navigation/poi_knowledge_base.py

key-decisions:
  - "Flat response shape {'waypoints': [...]} takes priority over nested {'markers': {'waypoints': [...]}} for protocol compatibility"
  - "isinstance(result, dict) guard in fetch_markers prevents AttributeError when AsyncMock returns default MagicMock (no return_value set)"
  - "Lowercase key normalization at load() time rather than at resolve() time for O(1) per-key lookup"

patterns-established:
  - "Defensive both-spelling check: result.get('avaliable_list', result.get('available_list', [])) for chassis protocol typos"
  - "update_chassis_markers() as explicit cache setter, called from fetch_markers and delete_marker"

requirements-completed:
  - POI-01
  - POI-02
  - POI-03
  - POI-04
  - SETUP-02

# Metrics
duration: 8min
completed: 2026-03-14
---

# Phase 10 Plan 03: POIKnowledgeBase Implementation Summary

**POIKnowledgeBase with full marker CRUD via chassis protocol, case-insensitive name resolution from per-floor JSON configs, and graceful missing-file handling via POI_CONFIG_MISSING event**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-14T07:43:00Z
- **Completed:** 2026-03-14T07:51:22Z
- **Tasks:** 1 (GREEN implementation)
- **Files modified:** 1

## Accomplishments

- Replaced all NotImplementedError stubs in poi_knowledge_base.py with full implementations
- All 5 tests (POI-01 through POI-04, SETUP-02) passing GREEN
- Case-insensitive name resolution with chassis marker fallback working correctly
- Per-floor JSON config loading with lowercase key normalization
- Defensive handling of "avaliable_list" protocol typo (checks both spellings)
- No regressions in chassis client tests (14 tests still passing)

## Task Commits

1. **Task 1: POIKnowledgeBase GREEN implementation** - `ffd757b` (feat)

## Files Created/Modified

- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/navigation/poi_knowledge_base.py` - Full implementation replacing skeleton stubs: load, resolve, list_locations, update_chassis_markers, fetch_markers, add_marker, delete_marker

## Decisions Made

- Used `isinstance(result, dict)` guard in `fetch_markers()` to handle the case where `AsyncMock` with no configured return value causes attribute errors in test_add_marker. This is also correct defensive coding for real network error cases.
- Normalized JSON config keys to lowercase at `load()` time rather than at `resolve()` time — keeps the resolve loop simple and consistent.
- Supported both flat `{"waypoints": [...]}` and nested `{"markers": {"waypoints": [...]}}` response shapes in `fetch_markers()` since test uses the flat form and the plan protocol description uses nested.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed AttributeError in fetch_markers when call_service returns non-dict**
- **Found during:** Task 1 (GREEN implementation, test_add_marker failure)
- **Issue:** test_add_marker does not set `call_service.return_value`, so AsyncMock returns a MagicMock when awaited. Calling `.get()` on a MagicMock is fine but `"waypoints" in MagicMock` triggered a coroutine-related warning and the fallback path hit an AttributeError.
- **Fix:** Added `if not isinstance(result, dict): return []` guard before parsing the response
- **Files modified:** smait/navigation/poi_knowledge_base.py
- **Verification:** test_add_marker passes; test_fetch_markers still returns correct waypoints list
- **Committed in:** ffd757b

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug fix for non-dict guard)
**Impact on plan:** Fix is strictly additive safety — handles both test mocks without return_value and real-world network error responses gracefully. No scope creep.

## Issues Encountered

- test_fetch_markers expected `call_service("/marker_operation/get_markers")` with no second argument (not `{"op": 0}` as the plan's protocol description implied). Matched the test assertion exactly. The response format in the test was also flat `{"waypoints": [...]}` rather than nested — implemented dual-shape parsing to support both.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- POIKnowledgeBase fully implemented and tested — ready for NavController (Plan 10-04) to use `resolve()` for destination lookup
- All POI requirements (POI-01 through POI-04) and SETUP-02 satisfied
- Phase 11 LLM tool layer can depend on `POIKnowledgeBase` for location name resolution

---
*Phase: 10-map-poi-and-navigation-server-code-home*
*Completed: 2026-03-14*
