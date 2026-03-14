---
phase: 11-wayfinding-llm-tools-and-display-rendering-home
plan: "02"
subsystem: connection
tags: [websocket, protocol, display, navigation, eventbus, tdd]

# Dependency graph
requires:
  - phase: 10-map-poi-and-navigation-server-code-home
    provides: EventBus, ConnectionManager base class with send_binary/send_text, BinaryFrame.pack pattern
provides:
  - FrameType.MAP_IMAGE = 0x06 in protocol.py
  - MessageSchema.nav_status() JSON constructor in protocol.py
  - ConnectionManager.send_map_image() sends 0x06 binary frame to Jackie
  - ConnectionManager.send_nav_status() sends nav_status JSON to Jackie
  - DISPLAY_MAP and DISPLAY_NAV_STATUS EventTypes in events.py
  - EventBus subscriptions wiring display events to WebSocket send methods
affects:
  - phase 11 plan 01 (WayfindingManager emits DISPLAY_MAP and DISPLAY_NAV_STATUS)
  - phase 12 (Android app receives 0x06 frames and nav_status JSON)
  - phase 13 (lab integration verification of end-to-end display dispatch)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD: RED skeleton with NotImplementedError stubs, GREEN minimal implementation"
    - "EventBus subscription in ConnectionManager.__init__ for display events"
    - "BinaryFrame.pack(FrameType.X, payload) pattern extended to MAP_IMAGE"

key-files:
  created:
    - tests/unit/test_display_dispatch.py
  modified:
    - smait/core/events.py
    - smait/connection/protocol.py
    - smait/connection/manager.py

key-decisions:
  - "_on_display_map guards against empty png_bytes before calling send_map_image (avoids sending empty 0x06 frame)"
  - "_on_display_nav_status guards isinstance(data, dict) for AsyncMock safety and real-world error resilience"
  - "DISPLAY_MAP and DISPLAY_NAV_STATUS added in this plan (idempotent — Plan 01 had not run yet)"

patterns-established:
  - "Display event handlers follow guard pattern: check data type + required field before forwarding"
  - "Protocol extension: add FrameType constant, add MessageSchema method, add send method + event handler pair"

requirements-completed: [DISP-01, DISP-02]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 11 Plan 02: Display Dispatch Summary

**ConnectionManager extended with send_map_image (0x06 binary frame) and send_nav_status (nav_status JSON) wired to DISPLAY_MAP and DISPLAY_NAV_STATUS EventBus subscriptions**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-14T18:37:48Z
- **Completed:** 2026-03-14T18:39:37Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments

- Added `FrameType.MAP_IMAGE = 0x06` and `MessageSchema.nav_status()` to protocol layer
- Added `DISPLAY_MAP` and `DISPLAY_NAV_STATUS` EventTypes to events.py (Plan 01 had not run yet)
- Implemented `ConnectionManager.send_map_image` and `send_nav_status` with EventBus subscriptions
- All 6 TDD tests pass GREEN, 12 existing connection/protocol tests have zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Display dispatch tests, protocol extension skeleton** - `d953a0d` (test)
2. **Task 2: GREEN -- Implement send_map_image, send_nav_status, and event handlers** - `ae082fa` (feat)

_Note: TDD tasks have RED (test) and GREEN (feat) commits._

## Files Created/Modified

- `smait/core/events.py` - Added DISPLAY_MAP and DISPLAY_NAV_STATUS EventTypes
- `smait/connection/protocol.py` - Added FrameType.MAP_IMAGE = 0x06 and MessageSchema.nav_status()
- `smait/connection/manager.py` - Added send_map_image, send_nav_status, _on_display_map, _on_display_nav_status with subscriptions
- `tests/unit/test_display_dispatch.py` - 6 new tests for DISP-01 and DISP-02 (all passing)

## Decisions Made

- **DISPLAY_MAP/NAV_STATUS added here, not Plan 01:** Plan 01 had not completed before this plan ran. Plan noted this case explicitly — added idempotently in Task 1.
- **Guard on empty png_bytes:** `_on_display_map` only calls `send_map_image` when png_bytes is truthy. Prevents sending a bare 0x06 frame with no payload.
- **isinstance(data, dict) guard:** `_on_display_nav_status` checks type before accessing keys, consistent with existing Pattern 10 decisions.

## Deviations from Plan

None - plan executed exactly as written (including the noted idempotent DISPLAY_MAP/DISPLAY_NAV_STATUS addition from the plan's NOTE section).

## Issues Encountered

None. The 2 protocol-level tests (test_frame_type_map_image, test_message_schema_nav_status) passed immediately in the RED phase because the plan's Task 1 action items explicitly required adding FrameType.MAP_IMAGE and MessageSchema.nav_status() before writing tests. This is expected and correct behavior.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Display dispatch layer is complete and tested
- Plan 11-01 (WayfindingManager) can emit DISPLAY_MAP and DISPLAY_NAV_STATUS events; they will be forwarded to Jackie automatically
- Phase 12 (Android app) needs to handle 0x06 binary frames and `nav_status` JSON messages
- Phase 13 lab integration can verify end-to-end: WayfindingManager → EventBus → ConnectionManager → Jackie touchscreen

---
*Phase: 11-wayfinding-llm-tools-and-display-rendering-home*
*Completed: 2026-03-14*
