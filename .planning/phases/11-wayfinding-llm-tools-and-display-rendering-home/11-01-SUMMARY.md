---
phase: 11-wayfinding-llm-tools-and-display-rendering-home
plan: "01"
subsystem: navigation
tags: [tdd, wayfinding, llm-tools, map-rendering, event-bus]
dependency_graph:
  requires:
    - "smait/navigation/map_manager.py (MapManager, world_to_pixel, draw_robot_arrow)"
    - "smait/navigation/poi_knowledge_base.py (POIKnowledgeBase.resolve, list_locations)"
    - "smait/navigation/nav_controller.py (NavController.navigate_to)"
    - "smait/core/events.py (EventBus, EventType)"
    - "smait/core/config.py (Config, NavigationConfig)"
  provides:
    - "WayfindingManager with get_tools() returning 2 OpenAI tool definitions"
    - "WayfindingManager._handle_query_location: resolves POI, emits DISPLAY_MAP"
    - "WayfindingManager._handle_navigate_to: triggers nav, emits DISPLAY_NAV_STATUS"
    - "MapManager.render_map_with_highlight: draws yellow ellipse at POI pixel"
    - "EventType.DISPLAY_MAP and DISPLAY_NAV_STATUS"
    - "NavigationConfig.highlight_color and highlight_radius_px"
  affects:
    - "smait/navigation/__init__.py (WayfindingManager now exported)"
    - "smait/navigation/map_manager.py (POI position cache, highlight rendering)"
tech_stack:
  added: []
  patterns:
    - "OpenAI function-calling tool definition format (type: function, function: {name, description, parameters})"
    - "EventBus.emit for side-effect dispatch from async tool handlers"
    - "PIL ImageDraw.ellipse for destination highlight circle overlay"
    - "POI position cache (_poi_positions dict) populated via POI_LIST_UPDATED event"
key_files:
  created:
    - smait/navigation/wayfinding_manager.py
    - tests/unit/test_wayfinding_manager.py
  modified:
    - smait/core/events.py
    - smait/core/config.py
    - smait/navigation/__init__.py
    - smait/navigation/map_manager.py
decisions:
  - "WayfindingManager owns DISPLAY_MAP and DISPLAY_NAV_STATUS emission (not MapManager or NavController)"
  - "MapManager._poi_positions cache populated via POI_LIST_UPDATED subscription in start() — avoids tight coupling with POIKnowledgeBase internals"
  - "render_map_with_highlight draws highlight before robot arrow so arrow renders on top"
  - "Supports both flat {x,y} and nested pose.position.{x,y} marker coordinate formats"
metrics:
  duration_s: 230
  tasks_completed: 2
  files_changed: 6
  completed_date: "2026-03-14"
requirements: [WAY-01, WAY-02, WAY-03, WAY-04]
---

# Phase 11 Plan 01: WayfindingManager and MapManager Highlight Summary

**One-liner:** WayfindingManager with query_location/navigate_to OpenAI tool definitions, POI-cached highlight rendering via MapManager, and DISPLAY_MAP/DISPLAY_NAV_STATUS EventType dispatch.

## What Was Built

### WayfindingManager (`smait/navigation/wayfinding_manager.py`)

New class providing LLM tool definitions and async handlers that bridge the LLM tool-call layer to the navigation/map subsystems.

- `get_tools()` returns `WAYFINDING_TOOLS` — two OpenAI function-calling tool definitions for `query_location` and `navigate_to`
- `get_tool_handlers()` returns `{"query_location": ..., "navigate_to": ...}` dict of async callables
- `_handle_query_location(args)`: calls `POIKnowledgeBase.resolve()`, renders highlighted map, emits `DISPLAY_MAP`, returns `{found, poi_name, verbal}`
- `_handle_navigate_to(args)`: calls `NavController.navigate_to()`, emits `DISPLAY_NAV_STATUS` on success, returns `{started, verbal}`

### MapManager extensions (`smait/navigation/map_manager.py`)

- Added `_poi_positions: dict[str, dict]` cache initialized in `__init__`
- Added `POI_LIST_UPDATED` subscription in `start()` to populate the cache
- Added `_on_poi_list_updated(data)`: extracts POI world coords from both flat `{x, y}` and nested `pose.position.{x, y}` marker formats
- Added `render_map_with_highlight(poi_name, highlight_color=None, radius=None)`: copies base image, draws yellow ellipse at POI pixel via `world_to_pixel`, then draws path and robot arrow

### EventType additions (`smait/core/events.py`)

```python
DISPLAY_MAP = auto()              # data: {"png": bytes, "highlighted_poi": str}
DISPLAY_NAV_STATUS = auto()       # data: {"status": str, "destination": str}
```

### NavigationConfig additions (`smait/core/config.py`)

```python
highlight_color: str = "yellow"
highlight_radius_px: int = 12
```

## Tests

9 tests in `tests/unit/test_wayfinding_manager.py`, all passing:

| Test | Requirement | What it verifies |
|------|-------------|-----------------|
| test_tool_registration | WAY-01 | get_tools() returns 2 tools with correct names |
| test_tool_handlers | WAY-01 | get_tool_handlers() returns async callables |
| test_query_location_found | WAY-01 | resolve() called, returns found=True, poi_name |
| test_query_location_not_found | WAY-01 | returns found=False with "I don't know" verbal |
| test_query_location_dispatches_map | WAY-03 | DISPLAY_MAP emitted with PNG bytes + highlighted_poi |
| test_navigate_to_success | WAY-02 | navigate_to called, returns started=True |
| test_navigate_to_failure | WAY-02 | returns started=False when nav fails |
| test_navigate_to_dispatches_status | WAY-04 | DISPLAY_NAV_STATUS emitted with status=navigating |
| test_render_map_highlight_pixel | WAY-03 | pixel at POI location is non-grey (highlight drawn) |

22 existing navigation regression tests continue to pass.

## Commits

| Hash | Description |
|------|-------------|
| c7cb471 | test(11-01): RED — wayfinding manager tests, EventType additions, config additions, class skeleton |
| 1e7c8e7 | feat(11-01): GREEN — implement WayfindingManager handlers and MapManager.render_map_with_highlight |

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- smait/navigation/wayfinding_manager.py — FOUND
- tests/unit/test_wayfinding_manager.py — FOUND
- Commit c7cb471 — FOUND
- Commit 1e7c8e7 — FOUND
