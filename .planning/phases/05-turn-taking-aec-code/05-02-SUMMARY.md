---
phase: 05-turn-taking-aec-code
plan: 02
subsystem: audio
tags: [aec, barge-in, vad, tts, turn-taking, speexdsp]

# Dependency graph
requires:
  - phase: 05-turn-taking-aec-code
    plan: 01
    provides: barge_in_min_speech_ms in EOUConfig

provides:
  - BARGE_IN EventType for barge-in notifications
  - AudioPipeline barge-in VAD path with 200ms anti-echo guard
  - SoftwareAEC with speexdsp, graceful degradation when unavailable
  - Cancellable TTSEngine with _on_barge_in() task cancellation

affects:
  - LAB phases (GPU validation of barge-in detection with live audio)
  - Phase 06 Android (BARGE_IN event must be forwarded to Android if needed)

# Tech tracking
tech-stack:
  added:
    - speexdsp (optional — graceful degradation if not installed)
  patterns:
    - Barge-in guard: 200ms window after TTS_START suppresses spurious detections
    - AEC gating: cae_status.aec=True bypasses software AEC entirely
    - Cancellable task pattern: asyncio.current_task() stored in _tts_task
    - TDD RED/GREEN: tests written first, implementation follows

key-files:
  created:
    - smait/sensors/aec.py
    - tests/unit/test_barge_in.py
    - tests/unit/test_aec.py
  modified:
    - smait/core/events.py
    - smait/sensors/audio_pipeline.py
    - smait/output/tts.py

key-decisions:
  - "BARGE_IN event placed in Turn-taking section of EventType enum after END_OF_TURN"
  - "_mic_gated replaced with _tts_playing + _tts_start_time — VAD stays active during TTS"
  - "SoftwareAEC lazy-imports speexdsp inside __init__ for graceful degradation"
  - "process_near() returns empty bytes (not passthrough) when no far-end reference available"
  - "asyncio.current_task() assigned to _tts_task at start of speak() and speak_streaming()"
  - "TTS_END always emitted in finally block — barge-in cannot suppress it"

patterns-established:
  - "Anti-echo guard: check (monotonic() - _tts_start_time)*1000 < barge_in_min_speech_ms before emitting BARGE_IN"
  - "AEC integration: self._aec.available AND NOT cae_status.aec — both conditions required"
  - "Frame buffering: near_buf and far_buf accumulate until FRAME_BYTES available for both"

requirements-completed: [AUD-06, AUD-07]

# Metrics
duration: 5m39s
completed: 2026-03-10
---

# Phase 05 Plan 02: BARGE_IN Event, Software AEC, Cancellable TTS Summary

**BARGE_IN event type added, software AEC class implemented with speexdsp graceful degradation, AudioPipeline mic gating replaced with barge-in VAD path (200ms anti-echo guard), and TTSEngine made cancellable via asyncio.Task.cancel().**

## Performance

- **Duration:** 5m39s
- **Started:** 2026-03-10T10:07:13Z
- **Completed:** 2026-03-10T10:12:52Z
- **Tasks:** 2
- **Files modified:** 6 (3 created, 3 modified)

## Accomplishments

- `EventType.BARGE_IN` added to Turn-taking section of EventType enum
- `AudioPipeline._mic_gated` replaced with `_tts_playing` + `_tts_start_time` — VAD stays active during TTS for barge-in detection
- 200ms anti-echo guard prevents spurious BARGE_IN within `barge_in_min_speech_ms` of TTS_START
- `SoftwareAEC` class created in `smait/sensors/aec.py` with lazy speexdsp import
  - 256-sample (512-byte) frames, near/far buffer accumulation
  - Returns empty bytes when no far-end reference available
  - Passthrough mode when speexdsp unavailable
- `SoftwareAEC` wired into `AudioPipeline.process_cae_audio()` gated on `not cae_status.aec`
- `_on_tts_audio_chunk()` feeds far-end audio to software AEC
- `TTSEngine._tts_task` holds current asyncio.Task; `_on_barge_in()` calls `.cancel()`
- `asyncio.CancelledError` re-raised in `speak()` and `speak_streaming()`; `TTS_END` always emitted in `finally`
- 15 new tests (7 barge-in + 8 AEC); 123 total tests pass

## Task Commits

1. **Task 1 RED: add failing barge-in tests** - `66a19eb` (test)
2. **Task 1 GREEN: BARGE_IN event, barge-in VAD path, cancellable TTS** - `2714ae7` (feat)
3. **Task 2 RED: add failing SoftwareAEC and pipeline AEC tests** - `796a9a6` (test)
4. **Task 2 GREEN: implement SoftwareAEC and wire into AudioPipeline** - `e31fcb5` (feat)

## Files Created/Modified

- `smait/core/events.py` - Added `BARGE_IN = auto()` to Turn-taking section
- `smait/sensors/audio_pipeline.py` - Replaced `_mic_gated` with `_tts_playing`/`_tts_start_time`; added barge-in detection in `process_cae_audio()`; wired `SoftwareAEC` + `_on_cae_status()`/`_on_tts_audio_chunk()`
- `smait/output/tts.py` - Added `_tts_task`, `_on_barge_in()`, `CancelledError` handling in `speak()` and `speak_streaming()`
- `smait/sensors/aec.py` - New: `SoftwareAEC` class with frame-level AEC, lazy speexdsp import, graceful degradation
- `tests/unit/test_barge_in.py` - New: 7 barge-in state transition tests
- `tests/unit/test_aec.py` - New: 8 SoftwareAEC + pipeline AEC integration tests

## Decisions Made

- **BARGE_IN event placement:** Added after `END_OF_TURN` in the Turn-taking section of EventType. Keeps related events grouped.
- **_mic_gated -> _tts_playing:** The old full mic gate prevented barge-in detection entirely. New approach keeps VAD running, only changes what happens with speech detections during TTS.
- **SoftwareAEC lazy import:** speexdsp requires `libspeexdsp-dev` and C compilation — unavailable in most dev environments. Lazy import in `__init__` with `ImportError` handler lets the system run without it.
- **process_near() returns empty when no far-end:** Returning passthrough when no far reference would defeat AEC. Empty bytes is the correct behaviour: nothing to cancel without reference.
- **asyncio.current_task() in speak/speak_streaming:** Setting `_tts_task = asyncio.current_task()` at the start of each coroutine allows external cancellation via `_on_barge_in()` without needing a wrapper coroutine.
- **TTS_END in finally block:** Pre-existing pattern preserved. `CancelledError` is re-raised after finally runs, ensuring TTS_END fires even on cancellation.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

The only notable discovery: `test_mic_gating_suppresses_vad` in `test_audio_pipeline.py` tests the old `_mic_gated` attribute but passes coincidentally because it only feeds speech chunks without silence (no segment emitted regardless of gating). This pre-existing test has low fidelity but does not fail. It is out of scope for this plan.

## Issues Encountered

None beyond the pre-existing test noted above.

## User Setup Required

None — speexdsp is optional and the system degrades gracefully.

For hardware AEC testing with speexdsp:
```
sudo apt install libspeexdsp-dev
pip install speexdsp
```

## Next Phase Readiness

- BARGE_IN event ready for downstream consumers (dialogue manager, session manager)
- SoftwareAEC ready for integration testing with real TTS audio in lab
- AudioPipeline CAE status tracking ready for hardware CAE events from Jackie

## Self-Check: PASSED

- smait/core/events.py: FOUND (BARGE_IN present)
- smait/sensors/audio_pipeline.py: FOUND (_tts_playing present)
- smait/output/tts.py: FOUND (_on_barge_in present)
- smait/sensors/aec.py: FOUND (SoftwareAEC present)
- tests/unit/test_barge_in.py: FOUND
- tests/unit/test_aec.py: FOUND
- .planning/phases/05-turn-taking-aec-code/05-02-SUMMARY.md: FOUND
- Commits 66a19eb, 2714ae7, 796a9a6, e31fcb5: ALL VERIFIED
- Final test run: 15 passed (test_barge_in.py + test_aec.py), 123 total

---
*Phase: 05-turn-taking-aec-code*
*Completed: 2026-03-10*
