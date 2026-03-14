---
phase: 09-chassis-websocket-client
plan: "02"
subsystem: chassis-client
tags:
  - chassis
  - websocket
  - tdd-green
  - implementation
dependency_graph:
  requires:
    - ChassisConfig dataclass in smait.core.config (Plan 01)
    - CHASSIS_* EventType members in smait.core.events (Plan 01)
    - 7 failing test cases from Plan 01 RED phase
  provides:
    - ChassisClient class in smait.connection.chassis_client
    - Full WebSocket connection, subscription, and message routing
    - send_soft_stop and call_service public API
    - 14 passing chassis tests covering CHAS-01 through CHAS-06
  affects:
    - smait/connection/chassis_client.py (created)
    - tests/unit/test_chassis_client.py (7 additional coverage tests added)
tech_stack:
  added:
    - pytest-cov (installed for coverage verification)
  patterns:
    - TDD GREEN phase — all 7 RED tests now pass
    - websockets asyncio connect iterator (auto-reconnect)
    - Dual fragment reassembly: rosbridge op=fragment AND raw partial-JSON buffering
    - asyncio.Event for connection state tracking
    - asyncio.Future for call_service RPC
key_files:
  created:
    - smait/connection/chassis_client.py
  modified:
    - tests/unit/test_chassis_client.py
decisions:
  - "Dual fragment handling: rosbridge op=fragment protocol AND raw partial-JSON buffer for two-frame split messages"
  - "event_bus property exposed publicly (client.event_bus) as tests access it directly"
  - "stop() cancels background task after closing WS to ensure clean shutdown"
  - "call_service uses asyncio.Future registered in _pending_calls, resolved by service_response handler"
metrics:
  duration_s: 300
  completed_date: "2026-03-13"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 9 Plan 02: ChassisClient Implementation Summary

ChassisClient fully implemented in TDD GREEN phase — WebSocket client with auto-reconnect, 4-topic subscription, bidirectional chassis protocol, rosbridge fragment reassembly, and 88% test coverage across 14 passing tests.

## What Was Built

### Task 1: ChassisClient Implementation

Created `smait/connection/chassis_client.py` (344 lines) with:

**Connection management:**
- `__init__`: stores `config.chassis`, `event_bus`, creates `asyncio.Event` for connection state, `itertools.count` for message IDs, `dict` for pending `call_service` futures, and fragment buffers
- `start()` / `stop()`: launch/cancel background `_run()` task; `stop()` closes WS and cancels the task
- `_run()`: uses `async for ws in connect(uri)` for automatic reconnect with exponential backoff from websockets library; emits `CHASSIS_CONNECTED` / `CHASSIS_DISCONNECTED` on state transitions; cancels all pending `call_service` futures on disconnect

**Subscriptions:**
- `_setup_subscriptions()`: sends subscribe ops for all 4 topics with correct ROS message types (`geometry_msgs/Pose2D`, `yutong_assistance/RobotStatus`, `actionlib_msgs/GoalStatus`, `std_msgs/Int8`)

**Message routing:**
- `_process_raw()`: tries JSON parse; buffers partial frames and retries on accumulation (handles the two-half-frame test pattern)
- `_handle_message()`: dispatches by `op` field to publish, fragment, or service_response handlers
- `_handle_publish()`: routes by topic to correct EventType emission with proper data shapes
- `_handle_fragment()`: rosbridge protocol — collects chunks by `id`/`num`/`total`, reassembles when complete
- `_handle_service_response()`: resolves pending `call_service` Future

**Event emissions:**
- `CHASSIS_POSE_UPDATE`: `{"x", "y", "theta"}` from pose topic
- `CHASSIS_NAV_STATUS`: `{"status", "text", "goal_id"}` with `NAV_STATUS_MAP` lookup
- `CHASSIS_STATE_UPDATE`: `{"battery", "nav_status", "control_state", "velocity"}`
- `CHASSIS_OBSTACLE`: `{"region"}`
- `CHASSIS_CONNECTED` / `CHASSIS_DISCONNECTED` on connection state changes

**Outgoing commands:**
- `send_soft_stop(stop: bool)`: advertise + publish ops for `/soft_stop` topic with `std_msgs/Bool`
- `call_service(service, args, timeout)`: async RPC with Future-based resolution and timeout cleanup

Module-level `NAV_STATUS_MAP`: `{0: "pending", 1: "active", 2: "preempted", 3: "succeeded", 4: "aborted"}`

### Task 2: Coverage Tests

Added 7 additional tests to `tests/unit/test_chassis_client.py`:

| Test | Coverage target |
|------|-----------------|
| `test_connected_property` | `connected` property before/after start/stop |
| `test_obstacle_subscription` | `CHASSIS_OBSTACLE` event emission |
| `test_send_soft_stop_when_disconnected` | RuntimeError when not connected |
| `test_call_service_when_disconnected` | RuntimeError when not connected |
| `test_call_service_timeout` | `asyncio.TimeoutError` with cleanup |
| `test_call_service_success` | Full RPC round-trip via `service_response` |
| `test_rosbridge_fragment_reassembly` | op=fragment multi-part protocol |

Final coverage: **88%** (164 stmts, 20 missed — all exception handler log lines)

## Verification Results

```
14 chassis tests: 14 passed
Full suite: 135 passed, 4 skipped, 6 failed (all pre-existing, unchanged)
chassis_client.py coverage: 88% (target: 80%)
ChassisClient importable: confirmed
```

## Decisions Made

1. **Dual fragment handling:** Tests use `push_fragments` which sends two raw partial-JSON strings as separate WebSocket frames. The plan described rosbridge `op=fragment`. Implemented both: `_process_raw` buffers raw partial JSON until valid, and `_handle_fragment` handles rosbridge protocol. Both test patterns pass.

2. **`event_bus` property:** Tests call `_get_event_bus(client)` which accesses `client.event_bus`. Exposed as a public property (not just `_bus`) to match test expectations cleanly.

3. **`stop()` cancels background task:** After setting `_running = False` and closing the WebSocket, `stop()` cancels and awaits the `_run()` task to ensure clean shutdown with no dangling coroutines.

4. **`call_service` Future cleanup:** On `TimeoutError`, the pending Future is popped from `_pending_calls` to prevent stale resolution if a late response arrives.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing functionality] Dual fragment handling for test compatibility**
- **Found during:** Task 1 verification
- **Issue:** Plan described rosbridge `op=fragment` protocol. Test's `push_fragments` sends two raw half-JSON strings as separate WebSocket frames — neither frame has `op=fragment`. Standard message routing (`json.loads`) would fail on each half.
- **Fix:** Added `_process_raw()` wrapper that buffers partial JSON and retries on each new frame. This handles both the test's raw-partial-frame pattern AND any raw protocol issues. Rosbridge `_handle_fragment` remains for the proper protocol.
- **Files modified:** `smait/connection/chassis_client.py`
- **Commit:** `a98465a`

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| smait/connection/chassis_client.py | FOUND |
| tests/unit/test_chassis_client.py | FOUND |
| 09-02-SUMMARY.md | FOUND |
| commit a98465a (Task 1 — ChassisClient) | FOUND |
| commit 2bc1d46 (Task 2 — coverage tests) | FOUND |
| All 14 chassis tests pass | CONFIRMED |
| chassis_client.py coverage 88% | CONFIRMED |
| No regressions in full suite | CONFIRMED |
