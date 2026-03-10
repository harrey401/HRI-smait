---
phase: 02-tts-pipeline-code
plan: 01
subsystem: tts
tags: [tts, streaming, emit-async, gpu-safety, pcm-conversion, tdd]
dependency_graph:
  requires: []
  provides: [TTS-01, TTS-02, TTS-03]
  affects: [smait/output/tts.py, tests/unit/test_tts.py]
tech_stack:
  added: []
  patterns: [emit_async ordered delivery, torch tensor guard, None audio guard]
key_files:
  created: []
  modified:
    - smait/output/tts.py
    - tests/unit/test_tts.py
decisions:
  - emit_async replaces emit() throughout TTSEngine — guarantees sequential delivery so TTS_END cannot race ahead of audio chunks
  - hasattr(audio, 'cpu') duck-type guard chosen over isinstance(audio, torch.Tensor) — avoids torch import in tts.py at all times
  - None audio guard placed before tensor guard — quiet mode check is cheapest and most common early exit
metrics:
  duration: ~8 minutes
  completed: "2026-03-10T07:14:26Z"
  tasks_completed: 1
  files_modified: 2
---

# Phase 02 Plan 01: TTS Emit Race Fix and GPU-Safe Conversion Summary

**One-liner:** emit_async replaces fire-and-forget emit() throughout TTSEngine with GPU-safe torch tensor handling and None audio skipping verified by 8 new unit tests.

## What Was Built

Fixed the TTSEngine race condition where `TTS_END` could fire before audio chunks finished delivering to Android's AudioTrack. Added GPU-safe audio conversion for KPipeline torch tensor output. Added quiet-mode None audio skipping.

### Changes to `smait/output/tts.py`

1. **`synthesize()`** — Added two guards before PCM conversion:
   - `if audio is None: continue` — skips KPipeline quiet-mode chunks
   - `if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()` — handles GPU torch tensors

2. **`speak()`** — All 3 `emit()` calls replaced with `await emit_async()` (TTS_START, TTS_AUDIO_CHUNK, TTS_END / DIALOGUE_RESPONSE)

3. **`_speak_by_sentence()`** — Both `emit()` calls replaced with `await emit_async()` (TTS_AUDIO_CHUNK, DIALOGUE_RESPONSE)

4. **`speak_streaming()`** — All 4 `emit()` calls replaced with `await emit_async()` (TTS_START, TTS_AUDIO_CHUNK x2, TTS_END). Removed unused `first_chunk` variable.

### Changes to `tests/unit/test_tts.py`

Added 8 new async test functions:

| Test | Verifies |
|------|----------|
| `test_pcm_conversion_correct` | float32 -> int16 with clipping at ±1.0 boundary |
| `test_torch_tensor_handled` | `.cpu().numpy()` called on torch-like mock tensors |
| `test_none_audio_skipped` | None chunks skipped, valid subsequent chunk processed |
| `test_audio_chunk_is_bytes` | `synthesize()` returns `bytes` (ConnectionManager contract) |
| `test_streaming_event_order` | TTS_START first, TTS_END last, AUDIO_CHUNKs in between |
| `test_streaming_emits_interleaved` | Sentence 1 chunk emitted before sentence 2 synthesized |
| `test_streaming_flushes_remainder` | Last sentence without trailing `.!?` flushed at stream end |
| `test_sentence_splitting` | "Hello. How? Fine." splits into exactly 3 synthesize calls |

## Test Results

```
14 passed in 3.80s
```

6 pre-existing tests + 8 new tests, all passing.

## Verification

- `grep -c "emit_async" smait/output/tts.py` → 10 (plan required 6+)
- `grep -c "\.emit(" smait/output/tts.py` → 0 (no fire-and-forget emit remaining)
- `grep "hasattr.*cpu" smait/output/tts.py` → confirmed present

## Commits

| Commit | Message |
|--------|---------|
| 9019692 | test(02-01): add failing RED tests for TTS streaming and conversion |
| 062d3b5 | feat(02-01): fix TTSEngine emit race condition and add GPU-safe audio conversion |

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `smait/output/tts.py` — modified, confirmed via grep checks
- `tests/unit/test_tts.py` — 14 tests collected and passing
- Commits 9019692 and 062d3b5 exist in git log
