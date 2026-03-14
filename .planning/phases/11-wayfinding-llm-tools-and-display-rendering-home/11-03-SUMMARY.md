---
phase: 11-wayfinding-llm-tools-and-display-rendering-home
plan: "03"
subsystem: dialogue
tags: [tdd, llm-tools, tool-calling, wayfinding, eventbus, openai]
dependency_graph:
  requires:
    - "smait/dialogue/manager.py (DialogueManager._ask_api, _history, _openai_client)"
    - "smait/navigation/wayfinding_manager.py (WayfindingManager from Plan 01)"
    - "smait/core/events.py (NAV_ARRIVED, NAV_FAILED, DIALOGUE_RESPONSE, DISPLAY_NAV_STATUS)"
  provides:
    - "DialogueManager.register_tools(tools, handlers) — stores tool defs and handlers"
    - "DialogueManager._ask_api passes tools=/tool_choice=auto to OpenAI when tools registered"
    - "DialogueManager._handle_tool_calls() — executes handlers, injects results, second LLM call"
    - "WayfindingManager._on_nav_arrived — emits DISPLAY_NAV_STATUS (arrived) + DIALOGUE_RESPONSE verbal"
    - "WayfindingManager._on_nav_failed — emits DISPLAY_NAV_STATUS (failed) + verbal explanation"
    - "WayfindingManager subscribes to NAV_ARRIVED and NAV_FAILED in __init__"
  affects:
    - "smait/navigation/wayfinding_manager.py (NAV event subscriptions and verbal handlers added)"
    - "smait/dialogue/manager.py (tool-call loop wired into _ask_api)"
tech_stack:
  added: []
  patterns:
    - "OpenAI two-round-trip tool-call flow: first call with tools=, second call with tool results"
    - "kwargs dict pattern for conditional tool parameter injection in _ask_api"
    - "WayfindingManager subscribes to nav outcome events at __init__ time"
    - "DialogueResponse emitted directly by WayfindingManager for verbal nav confirmations"
key_files:
  created: []
  modified:
    - smait/dialogue/manager.py
    - smait/navigation/wayfinding_manager.py
    - tests/unit/test_wayfinding_manager.py
decisions:
  - "Ollama path does NOT receive tools= parameter — local LLMs have unreliable tool-calling (per RESEARCH.md)"
  - "WayfindingManager subscribes to NAV_ARRIVED/NAV_FAILED in __init__ (not in a separate start() call)"
  - "DIALOGUE_RESPONSE emitted by WayfindingManager with model_used=wayfinding to distinguish from LLM responses"
  - "_handle_tool_calls uses kwargs dict pattern to conditionally add tools= (avoids if/else on create call)"
metrics:
  duration_s: 217
  tasks_completed: 2
  files_changed: 3
  completed_date: "2026-03-14"
requirements: [WAY-02, WAY-05]
---

# Phase 11 Plan 03: DialogueManager Tool-Call Loop and Nav Verbal Handlers Summary

**One-liner:** DialogueManager extended with register_tools() and two-round-trip OpenAI tool-call flow; WayfindingManager gets _on_nav_arrived/_on_nav_failed verbal confirmation handlers wired to NAV_ARRIVED/NAV_FAILED events.

## What Was Built

### DialogueManager tool registration (`smait/dialogue/manager.py`)

- Added `self._tools: list[dict] = []` and `self._tool_handlers: dict[str, Callable] = {}` to `__init__`
- Added `register_tools(tools, handlers)` — stores tool definitions and async handler callables
- Modified `_ask_api` to conditionally pass `tools=` and `tool_choice="auto"` via kwargs dict when tools are registered
- Added `_handle_tool_calls(message, messages)` — executes tool handlers, injects "tool" role results, makes second LLM call (no tools) to get verbal response
- Ollama (`_ask_ollama`) path unchanged — no tool support (per RESEARCH.md decision)

### WayfindingManager verbal handlers (`smait/navigation/wayfinding_manager.py`)

- Added `DialogueResponse` import from `smait.dialogue.manager`
- Added NAV_ARRIVED and NAV_FAILED subscriptions in `__init__`
- `_on_nav_arrived(data)`: emits `DISPLAY_NAV_STATUS` with `status=arrived`, then `DIALOGUE_RESPONSE` with "We've arrived at {destination}!"
- `_on_nav_failed(data)`: emits `DISPLAY_NAV_STATUS` with `status=failed`, then `DIALOGUE_RESPONSE` with "Sorry, I wasn't able to reach {destination}." plus optional reason

## Tests

5 new tests added to `tests/unit/test_wayfinding_manager.py`, all passing:

| Test | Requirement | What it verifies |
|------|-------------|-----------------|
| test_dialogue_register_tools | WAY-05 | register_tools() sets _tools and _tool_handlers |
| test_dialogue_tool_call_flow | WAY-05 | Two-round-trip: handler called, second OpenAI call made, second response returned |
| test_dialogue_no_tools_passthrough | WAY-05 | No tools= param when _tools is empty |
| test_nav_arrived_verbal | WAY-05 | NAV_ARRIVED fires DIALOGUE_RESPONSE + DISPLAY_NAV_STATUS arrived |
| test_nav_failed_verbal | WAY-05 | NAV_FAILED fires DIALOGUE_RESPONSE with "wasn't able to reach" + DISPLAY_NAV_STATUS failed |

**Total phase 11 tests:** 20 passing (9 Plan 01 + 6 Plan 02 + 5 Plan 03).

## Commits

| Hash | Description |
|------|-------------|
| 8eb5037 | test(11-03): RED — DialogueManager register_tools, tool-call flow, no-tools passthrough, nav arrived/failed verbal tests |
| ce947c5 | feat(11-03): GREEN — DialogueManager register_tools, _handle_tool_calls two-round-trip, WayfindingManager _on_nav_arrived/_on_nav_failed verbal handlers |

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- smait/dialogue/manager.py — FOUND (register_tools, _handle_tool_calls added)
- smait/navigation/wayfinding_manager.py — FOUND (_on_nav_arrived, _on_nav_failed added)
- tests/unit/test_wayfinding_manager.py — FOUND (5 new tests added)
- Commit 8eb5037 — FOUND
- Commit ce947c5 — FOUND
- All 20 phase 11 tests: PASSING
- wayfinding_manager.py coverage: 100%
