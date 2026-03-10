---
phase: 02-tts-pipeline-code
plan: "02"
subsystem: connection
tags: [tts, protocol, websocket, unit-tests, binary-frame]
dependency_graph:
  requires: []
  provides: [TTS-03-tests]
  affects: [smait/connection/protocol.py, smait/connection/manager.py]
tech_stack:
  added: []
  patterns: [TDD-green, AsyncMock, pytest-asyncio]
key_files:
  created:
    - tests/unit/test_protocol.py
    - tests/unit/test_connection_manager.py
  modified: []
decisions:
  - "Tests went straight to GREEN: production code (protocol.py, manager.py) already implemented TTS_AUDIO 0x05 wiring correctly — no bugs found"
  - "Added 2 extra tests beyond the 6 minimum (empty payload, dict-missing-audio-key, send_tts_audio-direct, no-client-no-raise) for fuller coverage"
metrics:
  duration: "2m13s"
  completed: "2026-03-10"
  tasks_completed: 1
  files_created: 2
  files_modified: 0
---

# Phase 02 Plan 02: TTS Protocol and ConnectionManager Tests Summary

Unit test coverage for the 0x05 TTS binary frame protocol and ConnectionManager audio forwarding — 12 tests across 2 files, all passing immediately (GREEN).

## What Was Built

Two new unit test files verifying the binary frame protocol contract and WebSocket forwarding:

- `tests/unit/test_protocol.py` — 6 tests for `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm)` encoding, roundtrip, and error cases
- `tests/unit/test_connection_manager.py` — 6 tests for `ConnectionManager._on_tts_audio_chunk` forwarding TTS events to a mocked WebSocket as 0x05 frames

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create protocol and ConnectionManager TTS tests | f1e9fb6 | tests/unit/test_protocol.py, tests/unit/test_connection_manager.py |

## Test Coverage

### test_protocol.py (6 tests)
- `test_tts_audio_frame_encoding` — `pack(TTS_AUDIO, b'\x01\x02\x03')` == `bytes([0x05, 0x01, 0x02, 0x03])`
- `test_tts_audio_frame_roundtrip` — `from_bytes(pack(TTS_AUDIO, payload))` reconstructs type and payload
- `test_binary_frame_too_short` — `from_bytes(b'\x05')` raises `ValueError`
- `test_tts_audio_frame_type_value` — `FrameType.TTS_AUDIO == 0x05`
- `test_tts_audio_empty_payload` — pack with empty bytes produces single `0x05` byte
- `test_binary_frame_too_short_empty` — `from_bytes(b'')` raises `ValueError`

### test_connection_manager.py (6 tests)
- `test_tts_audio_forwarded` — dict `{"audio": pcm}` payload sends `0x05 + pcm` to WebSocket
- `test_tts_audio_raw_bytes_forwarded` — raw bytes payload sends `0x05 + pcm` to WebSocket
- `test_tts_audio_no_client` — `None` client does not raise on TTS_AUDIO_CHUNK event
- `test_tts_audio_dict_missing_audio_key` — dict without `"audio"` key does not send to WebSocket
- `test_send_tts_audio_directly` — `send_tts_audio(pcm)` packs and sends `bytes([0x05]) + pcm`
- `test_send_tts_audio_no_client_no_raise` — `send_tts_audio()` with no client does not raise

## Verification

```
pytest tests/unit/test_protocol.py -v        → 6 passed
pytest tests/unit/test_connection_manager.py -v → 6 passed
```

## Deviations from Plan

None — plan executed exactly as written. Production code was already correct; tests went straight to GREEN. Two additional tests were added beyond the 6 minimum to improve edge-case coverage (dict without audio key, send_tts_audio direct call).

## Self-Check

- [x] `tests/unit/test_protocol.py` exists and has 6 passing tests
- [x] `tests/unit/test_connection_manager.py` exists and has 6 passing tests
- [x] Commit `f1e9fb6` verified in git log

## Self-Check: PASSED
