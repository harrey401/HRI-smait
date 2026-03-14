---
phase: 10-map-poi-and-navigation-server-code-home
plan: "01"
subsystem: navigation
tags: [navigation, map, poi, events, config, chassis-client, tdd-red]
dependency_graph:
  requires: [09-02]
  provides: [10-02, 10-03, 10-04]
  affects: [smait/core/events.py, smait/core/config.py, smait/connection/chassis_client.py]
tech_stack:
  added: [PIL (Pillow) for map_manager skeleton]
  patterns: [TDD RED phase, skeleton classes with NotImplementedError, EventBus pub/sub]
key_files:
  created:
    - smait/navigation/__init__.py
    - smait/navigation/map_manager.py
    - smait/navigation/poi_knowledge_base.py
    - smait/navigation/nav_controller.py
    - tests/unit/test_map_manager.py
    - tests/unit/test_poi_knowledge_base.py
    - tests/unit/test_nav_controller.py
    - data/poi/tefa/3.json
  modified:
    - smait/core/events.py
    - smait/core/config.py
    - smait/connection/chassis_client.py
decisions:
  - NavigationConfig stored as separate dataclass (not merged into ChassisConfig) for clean separation
  - _nav_cfg stored alongside _cfg in ChassisClient.__init__ for consistent access pattern
  - /global_path subscription added to _setup_subscriptions (auto-subscribed) not subscribe_topic (manual)
  - subscribe_topic is manual API for MapManager to use with compression/fragment_size kwargs
  - test_nav_status_events uses AssertionError (correct RED — skeleton has no event wiring yet)
metrics:
  duration_seconds: 260
  completed_date: "2026-03-14"
  tasks_completed: 2
  tasks_total: 2
  files_created: 8
  files_modified: 3
---

# Phase 10 Plan 01: Core Infrastructure and Navigation Skeletons Summary

**One-liner:** Extended EventBus with 11 navigation events, added NavigationConfig dataclass, extended ChassisClient with op:png handler and 3 new command methods, and created 3 navigation skeleton classes with 16 RED-state tests.

## What Was Built

### Task 1: Core Infrastructure

**smait/core/events.py** — 11 new EventType members under `# Map / Navigation (Phase 10)`:
- `CHASSIS_MAP_UPDATE`, `CHASSIS_PATH_UPDATE` — chassis telemetry
- `MAP_RENDERED`, `MAP_ACTIVE_FLOOR`, `MAP_LIST_UPDATED` — map lifecycle
- `NAV_STARTED`, `NAV_ARRIVED`, `NAV_FAILED`, `NAV_CANCELLED` — navigation lifecycle
- `POI_LIST_UPDATED`, `POI_CONFIG_MISSING` — POI management

**smait/core/config.py** — New `NavigationConfig` dataclass with 10 fields (poi_config_dir, map_fragment_size, map_throttle_rate_ms, path_topic, map_topic, insert_marker_topic, cancel_nav_topic, arrow_color, path_color, arrow_length_px). Added `navigation: NavigationConfig` field to `Config`.

**smait/connection/chassis_client.py** — 5 extensions:
- `_nav_cfg` stored in `__init__` alongside `_cfg`
- `_handle_message` routes `op:png` to new `_handle_png` handler
- `_handle_png` emits `CHASSIS_MAP_UPDATE` with raw msg dict
- `_handle_publish` handles `/global_path` topic → `CHASSIS_PATH_UPDATE`
- `/global_path` subscription added to `_setup_subscriptions` with `yutong_assistance/point_array`
- New public methods: `subscribe_topic`, `send_cancel_navigation`, `send_insert_marker`

### Task 2: Navigation Module Skeletons + RED Tests

**smait/navigation/__init__.py** — Module init exporting MapManager, POIKnowledgeBase, NavController.

**smait/navigation/map_manager.py** — MapManager skeleton with 5 async methods (start, list_maps, switch_map, get_active_map_info, render_map) plus module-level helper stubs `world_to_pixel` and `draw_robot_arrow`.

**smait/navigation/poi_knowledge_base.py** — POIKnowledgeBase skeleton with 7 methods (load, resolve, list_locations, fetch_markers, add_marker, delete_marker, update_chassis_markers).

**smait/navigation/nav_controller.py** — NavController skeleton with 3 async methods (navigate_to, cancel_navigation, calculate_distance).

**data/poi/tefa/3.json** — Sample POI config for TEFA engineering lab floor 3 (5 entries: room ENG192, bathroom, elevator, lab entrance, professor office).

**16 test functions in RED state:**
- `tests/unit/test_map_manager.py` — 7 tests (MAP-01 to MAP-04, NAV-02, SETUP-03)
- `tests/unit/test_poi_knowledge_base.py` — 5 tests (POI-01 to POI-04, SETUP-02)
- `tests/unit/test_nav_controller.py` — 4 tests (NAV-01, NAV-03, NAV-04, NAV-05)

## Verification Results

- `from smait.navigation import MapManager, POIKnowledgeBase, NavController` — PASS
- 16 tests collected, all failing with NotImplementedError (RED state confirmed)
- 14 existing chassis tests still passing (no regressions)
- `hasattr(EventType, 'CHASSIS_MAP_UPDATE')` — PASS
- `Config().navigation.poi_config_dir == 'data/poi'` — PASS
- `data/poi/tefa/3.json` valid JSON — PASS

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

Files created:
- smait/navigation/__init__.py — FOUND
- smait/navigation/map_manager.py — FOUND
- smait/navigation/poi_knowledge_base.py — FOUND
- smait/navigation/nav_controller.py — FOUND
- tests/unit/test_map_manager.py — FOUND
- tests/unit/test_poi_knowledge_base.py — FOUND
- tests/unit/test_nav_controller.py — FOUND
- data/poi/tefa/3.json — FOUND

Commits:
- 1b40562 — feat(10-01): extend core infra — FOUND
- be52eb3 — test(10-01): navigation module skeletons + RED test suites — FOUND
