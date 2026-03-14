---
phase: 09-chassis-websocket-client
plan: "01"
subsystem: chassis-client
tags:
  - chassis
  - websocket
  - config
  - events
  - tdd-red
dependency_graph:
  requires: []
  provides:
    - ChassisConfig dataclass in smait.core.config
    - 6 CHASSIS_* EventType members in smait.core.events
    - MockChassisServer fixture for Plan 02
    - 7 failing test cases for CHAS-01 through CHAS-06
  affects:
    - smait/core/config.py
    - smait/core/events.py
    - tests/unit/test_chassis_client.py
tech_stack:
  added:
    - websockets==16.0 (used in MockChassisServer fixture)
  patterns:
    - TDD RED phase — tests collected but fail with ImportError
    - deferred-import pattern allows pytest collection while ChassisClient is absent
key_files:
  created:
    - tests/unit/test_chassis_client.py
  modified:
    - smait/core/config.py
    - smait/core/events.py
decisions:
  - "Deferred ChassisClient import (try/except) so pytest collects 7 tests while RED; _require_chassis_client() raises ImportError at test runtime"
  - "Used websockets 16.0 asyncio serve API with port=0 for random OS-assigned test port"
metrics:
  duration_s: 199
  completed_date: "2026-03-13"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
---

# Phase 9 Plan 01: Chassis WebSocket Contracts Summary

ChassisConfig dataclass, 6 chassis EventType members, and a 7-test failing test suite with MockChassisServer fixture — establishing all contracts for the CHAS-01 through CHAS-06 requirements in TDD RED state.

## What Was Built

### Task 1: Config and Events

Added `ChassisConfig` dataclass to `smait/core/config.py` with 8 fields:
- `host` (default: `192.168.20.22`), `port` (default: `9090`)
- `reconnect_max_wait_s`, `pose_topic`, `status_topic`, `nav_status_topic`, `obstacle_topic`, `soft_stop_topic`

Added `chassis: ChassisConfig` field to the `Config` dataclass after the `logging` field.

Added 6 new `EventType` members in `smait/core/events.py` under a `# Chassis / Navigation` section:
- `CHASSIS_POSE_UPDATE` — data: `{"x": float, "y": float, "theta": float}`
- `CHASSIS_NAV_STATUS` — data: `{"status": int, "text": str, "goal_id": str}`
- `CHASSIS_STATE_UPDATE` — data: `{"battery": float, "nav_status": int, "control_state": int, "velocity": list}`
- `CHASSIS_OBSTACLE` — data: `{"region": int}`
- `CHASSIS_CONNECTED` — data: None
- `CHASSIS_DISCONNECTED` — data: None

### Task 2: MockChassisServer and Failing Tests

Created `tests/unit/test_chassis_client.py` with:
- `MockChassisServer` class — real WebSocket server, stores `received` messages, supports `push()`, `push_fragments()`, `close()`, `wait_connected()`
- `mock_chassis` pytest fixture — starts server on OS-assigned port, yields `(mock, port)`
- `chassis_client` fixture skeleton — ready for Plan 02 to use once `ChassisClient` is implemented
- 7 async test functions covering CHAS-01 through CHAS-06 plus fragment reassembly

All 7 tests are collected by pytest and fail with `ImportError` (the expected RED state — `smait.connection.chassis_client` does not exist yet).

## Decisions Made

1. **Deferred import pattern:** `ChassisClient` import uses `try/except ImportError` so pytest collects all 7 tests even though the module doesn't exist. A `_require_chassis_client()` guard at the top of each test raises `ImportError` at runtime with a clear message for Plan 02.

2. **websockets 16 asyncio API:** Used `websockets.asyncio.server.serve` with `port=0` for OS-assigned random ports, consistent with websockets 16.0 available in the project venv.

3. **nav_status text mapping:** Tests assert that `status=3` → `text="succeeded"`. This documents the expected mapping contract Plan 02 must implement.

## Verification Results

```
Config OK — c.chassis.host='192.168.20.22', c.chassis.port=9090
Events OK — 6 CHASSIS_* members present
Tests — 7 collected, 7 failed (ImportError — expected RED state)
Pre-existing tests — 116 passed, 5 pre-existing failures (unchanged)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Deferred import to allow pytest collection**
- **Found during:** Task 2
- **Issue:** Plan specified `from smait.connection.chassis_client import ChassisClient` at module level, which causes an `ImportError` at collection time, preventing pytest from collecting any tests (0 collected). Plan simultaneously required "7 test cases collected" and "fail with ImportError" — contradictory with a top-level import.
- **Fix:** Changed to `try/except ImportError` with `ChassisClient = None` fallback; added `_require_chassis_client()` guard at top of each test that raises `ImportError` at runtime.
- **Files modified:** `tests/unit/test_chassis_client.py`
- **Commit:** `010fe6a`

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| smait/core/config.py | FOUND |
| smait/core/events.py | FOUND |
| tests/unit/test_chassis_client.py | FOUND |
| 09-01-SUMMARY.md | FOUND |
| commit cc74ce6 (Task 1) | FOUND |
| commit 010fe6a (Task 2) | FOUND |
