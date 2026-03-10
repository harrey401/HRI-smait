---
phase: 02-tts-pipeline-code
verified: 2026-03-10T08:00:00Z
status: passed
score: 8/8 must-haves verified
re_verification: false
---

# Phase 02: TTS Pipeline Code Verification Report

**Phase Goal:** Kokoro TTS wrapper rewritten with correct KPipeline API and sentence-level streaming
**Verified:** 2026-03-10T08:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                        | Status     | Evidence                                                                                                 |
|----|----------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| 1  | TTSEngine uses KPipeline(lang_code='a') generator API, yielding (graphemes, phonemes, audio) | VERIFIED   | `tts.py:55` — `KPipeline(lang_code="a")`; `synthesize()` iterates `(graphemes, phonemes, audio)` tuples |
| 2  | Sentence-level streaming: first sentence audio emitted before full response synthesized       | VERIFIED   | `test_streaming_emits_interleaved` passes; `speak_streaming()` emits AUDIO_CHUNK per sentence in loop   |
| 3  | TTS audio encoded as 0x05 binary frames for WebSocket transmission                           | VERIFIED   | `protocol.py:91-93` — `BinaryFrame.pack` = `bytes([frame_type]) + payload`; TTS_AUDIO = 0x05           |
| 4  | Unit tests verify streaming behavior with mocked KPipeline                                   | VERIFIED   | 26 tests pass across 3 files; 8 streaming/conversion tests in `test_tts.py`                             |
| 5  | speak_streaming() emits sentence 1 audio BEFORE synthesizing sentence 2                      | VERIFIED   | `test_streaming_emits_interleaved` asserts EMIT_CHUNK index < SYNTHESIZE_2 index                        |
| 6  | All audio chunks arrive at Android before TTS_END signal                                     | VERIFIED   | `emit_async` used throughout; TTS_END only emitted in `finally` block after all chunks; `test_streaming_event_order` confirms order |
| 7  | TTS_AUDIO_CHUNK event data contains bytes, not raw tensors or arrays                         | VERIFIED   | `test_audio_chunk_is_bytes` passes; `synthesize()` returns `bytes` via `pcm.tobytes()`                  |
| 8  | synthesize() handles torch tensors via .cpu().numpy() guard and skips None audio             | VERIFIED   | `tts.py:90-96` — None guard then `hasattr(audio, 'cpu')` guard; confirmed by tests                      |

**Score:** 8/8 truths verified

---

### Required Artifacts

| Artifact                              | Expected                                            | Status    | Details                                                                  |
|---------------------------------------|-----------------------------------------------------|-----------|--------------------------------------------------------------------------|
| `smait/output/tts.py`                 | TTSEngine with emit_async streaming, GPU-safe audio | VERIFIED  | 204 lines; 10 `emit_async` calls; 0 bare `.emit(` calls; both guards present |
| `tests/unit/test_tts.py`             | 8+ streaming/conversion tests, min 140 lines        | VERIFIED  | 327 lines; 14 tests (6 pre-existing + 8 new); all pass                   |
| `tests/unit/test_protocol.py`        | BinaryFrame pack/parse tests, min 20 lines          | VERIFIED  | 42 lines; 6 tests covering encode, roundtrip, error cases; all pass       |
| `tests/unit/test_connection_manager.py` | ConnectionManager TTS forwarding tests, min 30 lines | VERIFIED | 84 lines; 6 tests covering dict/bytes payloads, no-client safety; all pass |

---

### Key Link Verification

| From                         | To                          | Via                                     | Status   | Details                                                                                      |
|------------------------------|-----------------------------|-----------------------------------------|----------|----------------------------------------------------------------------------------------------|
| `smait/output/tts.py`        | `smait/core/events.py`      | `emit_async` for TTS_AUDIO_CHUNK, TTS_START, TTS_END | WIRED | 10 `emit_async` calls confirmed in tts.py; 0 bare `emit()` calls remain                |
| `smait/connection/manager.py`| `smait/connection/protocol.py` | `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)` | WIRED | `manager.py:193` — `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)` confirmed   |
| `ConnectionManager.__init__` | `_on_tts_audio_chunk`       | `event_bus.subscribe(TTS_AUDIO_CHUNK, ...)`    | WIRED | `manager.py:41` — subscribe call exists; forwarding verified by 3 connection manager tests   |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                         | Status    | Evidence                                                                           |
|-------------|------------|---------------------------------------------------------------------|-----------|------------------------------------------------------------------------------------|
| TTS-01      | 02-01-PLAN  | Kokoro-82M TTS with `KPipeline(lang_code='a')` generator API        | SATISFIED | `tts.py:55`: `KPipeline(lang_code="a")`; `test_correct_class_imported` passes     |
| TTS-02      | 02-01-PLAN  | Sentence-level streaming TTS (yield per sentence, not per word)      | SATISFIED | `speak_streaming()` synthesizes per sentence boundary; 3 streaming tests verify    |
| TTS-03      | 02-01-PLAN, 02-02-PLAN | TTS audio sent as 0x05 binary frames to Android            | SATISFIED | `FrameType.TTS_AUDIO = 0x05`; `BinaryFrame.pack` prefixes 0x05; 9 tests verify   |

No orphaned requirements: TTS-01, TTS-02, TTS-03 all mapped to Phase 2 in REQUIREMENTS.md and covered by plans. REQUIREMENTS.md marks all three as `[x]` (complete).

---

### Anti-Patterns Found

| File                    | Line | Pattern | Severity | Impact |
|-------------------------|------|---------|----------|--------|
| None                    | —    | —       | —        | —      |

No TODOs, FIXMEs, placeholders, empty implementations, or console.log-only stubs found in any modified file (`smait/output/tts.py`, `smait/connection/manager.py`, `smait/connection/protocol.py`, `tests/unit/test_tts.py`, `tests/unit/test_protocol.py`, `tests/unit/test_connection_manager.py`).

---

### Human Verification Required

None. All success criteria are programmatically verifiable via unit tests, grep checks, and static analysis. The 26 passing tests provide strong confidence in correctness without needing runtime TTS model inference.

---

### Test Run Summary

```
26 passed in 3.86s
  tests/unit/test_tts.py            — 14 passed
  tests/unit/test_protocol.py       — 6 passed
  tests/unit/test_connection_manager.py — 6 passed
```

---

### Commit Verification

| Commit  | Message                                                        | Verified |
|---------|----------------------------------------------------------------|----------|
| 9019692 | test(02-01): add failing RED tests for TTS streaming and conversion | Yes |
| 062d3b5 | feat(02-01): fix TTSEngine emit race condition and add GPU-safe audio conversion | Yes |
| f1e9fb6 | test(02-02): add TTS 0x05 frame protocol and ConnectionManager forwarding tests | Yes |

All 3 commits confirmed present in git log with expected file changes.

---

### Phase Goal Narrative

The phase goal "Kokoro TTS wrapper rewritten with correct KPipeline API and sentence-level streaming" is fully achieved:

1. **KPipeline API**: `init_model()` instantiates `KPipeline(lang_code='a')` and `synthesize()` consumes the `(graphemes, phonemes, audio)` generator — replacing any prior stub API usage.

2. **Sentence-level streaming**: `speak_streaming()` buffers LLM tokens, detects sentence boundaries via `SENTENCE_BOUNDARY` regex, synthesizes each sentence independently, and emits audio before the next sentence begins. The flush path handles trailing text without a terminating punctuation-space sequence.

3. **0x05 binary frames**: `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm)` produces the `0x05` prefix byte + PCM payload. `ConnectionManager._on_tts_audio_chunk` subscribes to `TTS_AUDIO_CHUNK` events and forwards both dict-keyed and raw-bytes payloads via `send_tts_audio`.

4. **Race condition fix**: All `emit()` calls in `speak()`, `speak_streaming()`, and `_speak_by_sentence()` replaced with `await emit_async()`. `emit_async` uses `asyncio.gather` to await all handlers sequentially before continuing, preventing `TTS_END` from racing ahead of audio chunks.

5. **GPU safety**: `synthesize()` handles torch tensors via `hasattr(audio, 'cpu')` duck-type guard and skips `None` audio chunks from KPipeline quiet mode.

---

_Verified: 2026-03-10T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
