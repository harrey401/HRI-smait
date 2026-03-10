# Phase 2: TTS Pipeline Code - Research

**Researched:** 2026-03-09
**Domain:** Kokoro KPipeline streaming TTS, WebSocket binary frame protocol, sentence-level streaming patterns
**Confidence:** HIGH (KPipeline API read directly from installed source; protocol verified from `smait/connection/protocol.py`)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TTS-01 | Kokoro-82M TTS integrated with correct API (`KPipeline(lang_code='a')` generator) | KPipeline API confirmed by reading installed `venv/.../kokoro/pipeline.py` directly; `__call__` yields `KPipeline.Result` with backward-compat `__iter__` yielding `(graphemes, phonemes, audio)` |
| TTS-02 | Sentence-level streaming TTS (yield per sentence, not per word) | KPipeline yields per-chunk already; `speak_streaming()` buffer logic exists in `smait/output/tts.py` but lacks `emit_async` and needs sentence boundary regex to match `[.!?]` followed by whitespace OR end-of-string |
| TTS-03 | TTS audio sent as 0x05 binary frames to Android for AudioTrack playback | `FrameType.TTS_AUDIO = 0x05` defined in `smait/connection/protocol.py`; `ConnectionManager.send_tts_audio()` calls `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)`; `_on_tts_audio_chunk` subscribed to `EventType.TTS_AUDIO_CHUNK` — protocol wiring is already complete |
</phase_requirements>

---

## Summary

Phase 1 already corrected the Kokoro stub: `smait/output/tts.py` now uses `KPipeline(lang_code='a')` and iterates the `(graphemes, phonemes, audio)` generator in `synthesize()`. The 0x05 binary frame wiring is also complete in `ConnectionManager`. Phase 2 is therefore a **depth phase** — the scaffolding exists, but the streaming behavior is incomplete or undertested.

The key gaps are: (1) `speak_streaming()` accumulates whole-sentence text before calling `synthesize()`, which is correct, but it uses `self._event_bus.emit()` (fire-and-forget) rather than `emit_async()`, so audio chunks may race with TTS_END; (2) the sentence boundary regex `(?<=[.!?])\s+` misses the last sentence fragment that ends without whitespace (e.g., "Hello world"); (3) no tests verify the streaming interleaving behavior — that the first sentence audio is emitted before the second sentence is synthesized; and (4) the 0x05 framing is never tested directly against `TTSEngine`.

The KPipeline `__call__` method is a synchronous generator. It uses `split_pattern=r'\n+'` to segment text into paragraphs, then internally chunks by phoneme count (510 limit), yielding one `Result` per chunk. Each chunk is typically one sentence or sentence fragment. The backward-compat `__iter__` on `KPipeline.Result` makes tuple unpacking `for gs, ps, audio in pipeline(text, ...)` work — `audio` is `result.output.audio`, a `torch.FloatTensor`. It is `None` when `model=False` (quiet pipeline).

**Primary recommendation:** The existing `TTSEngine` structure is correct. Phase 2 work is: fix the sentence-boundary flush regex, switch to `emit_async` inside `speak_streaming`, and write tests that verify streaming interleave order with a mocked KPipeline.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| kokoro | 0.9.4 (installed) | KPipeline TTS generator | Already installed; `KPipeline.__call__` yields per-chunk audio |
| numpy | >=1.24.0 | PCM float32-to-int16 conversion | `(audio * 32767).clip(-32768, 32767).astype(np.int16)` — standard pattern in codebase |
| pytest | >=7.0.0 | Test runner | Configured in `pyproject.toml`; `asyncio_mode = "auto"` |
| pytest-asyncio | >=0.21.0 | Async test support | Enabled; all async tests in project already use it |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | Mock KPipeline without loading model weights | All unit tests — never load real model in unit tests |
| re | stdlib | Sentence boundary detection | `SENTENCE_BOUNDARY` regex in `tts.py` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom sentence splitter regex | NLTK sentence tokenizer | NLTK adds a dependency; regex is sufficient for 1-3 spoken sentences from LLM output |
| `event_bus.emit()` | `event_bus.emit_async()` | `emit()` schedules tasks on loop but does not await them; `emit_async()` awaits all handlers — required for ordered delivery |

**Installation:** No new packages required. All dependencies installed in Phase 1.

---

## Architecture Patterns

### Recommended Project Structure

```
smait/output/
└── tts.py          # TTSEngine — all changes are here

tests/unit/
└── test_tts.py     # Extend existing test file with streaming tests
```

### Pattern 1: KPipeline Generator Iteration

**What:** `KPipeline.__call__` is a synchronous generator that yields `KPipeline.Result` objects. Backward-compat `__iter__` on `Result` enables tuple unpacking as `(graphemes, phonemes, audio)`. The `audio` field is `result.output.audio` — a `torch.FloatTensor` at 24kHz. Convert to PCM16 bytes with `(audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()`.

**When to use:** In `synthesize(text)` — call `self._pipeline(text, voice=self._voice, speed=1.0)` and iterate.

**Verified from:** `/home/gow/.openclaw/workspace/projects/SMAIT-v3/venv/lib/python3.12/site-packages/kokoro/pipeline.py` lines 338-349 (backward-compat `__iter__`) and lines 351-432 (`__call__`).

```python
# Source: installed kokoro/pipeline.py (v0.9.4), lines 338-349, 351-432
async def synthesize(self, text: str) -> Optional[bytes]:
    if not self._available or self._pipeline is None:
        return None
    try:
        t0 = time.monotonic()
        pcm_parts = []
        for _graphemes, _phonemes, audio in self._pipeline(
            text, voice=self._voice, speed=1.0
        ):
            # audio is torch.FloatTensor at 24kHz; backward-compat __iter__ yields it as 3rd element
            # audio may be None if pipeline was constructed with model=False (quiet mode)
            if audio is None:
                continue
            # Convert to numpy if needed (KModel.Output.audio is a torch tensor)
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_parts.append(pcm.tobytes())
        pcm_bytes = b"".join(pcm_parts)
        latency = (time.monotonic() - t0) * 1000
        logger.debug("TTS: '%s' -> %d bytes (%.1fms)", text[:50], len(pcm_bytes), latency)
        return pcm_bytes if pcm_bytes else None
    except Exception:
        logger.exception("TTS synthesis failed")
        return None
```

### Pattern 2: Sentence-Level Streaming with `emit_async`

**What:** `speak_streaming()` consumes an `AsyncGenerator[str, None]` from the LLM, buffers tokens, detects sentence boundaries, synthesizes and emits audio per sentence — all before the full LLM response is done. The critical fix is using `emit_async()` instead of `emit()` to ensure audio chunks are forwarded to the WebSocket before TTS_END is emitted.

**When to use:** Called by `DialogueManager` with a streaming LLM generator. This is the primary path for the conversation loop.

**Sentence boundary regex issue:** The current regex `(?<=[.!?])\s+` requires whitespace after the punctuation. The last sentence in a fragment (e.g., "Hello world.") never matches because the buffer ends with a period and no trailing whitespace. Fix: flush remaining buffer as a sentence at stream end (already present) AND also trigger a flush on `[.!?]` followed by end-of-string when the buffer exceeds a minimum length.

```python
# Source: analysis of smait/output/tts.py + smait/core/events.py
# Key pattern: use emit_async to ensure ordered delivery
async def speak_streaming(self, text_generator: AsyncGenerator[str, None]) -> None:
    self._is_speaking = True
    await self._event_bus.emit_async(EventType.TTS_START)

    self._text_buffer = ""
    try:
        async for chunk in text_generator:
            self._text_buffer += chunk
            # Extract complete sentences from buffer
            while True:
                match = SENTENCE_BOUNDARY.search(self._text_buffer)
                if not match:
                    break
                sentence = self._text_buffer[:match.start()].strip()
                self._text_buffer = self._text_buffer[match.end():]
                if sentence:
                    pcm = await self.synthesize(sentence)
                    if pcm:
                        await self._event_bus.emit_async(
                            EventType.TTS_AUDIO_CHUNK, {"audio": pcm}
                        )
        # Flush remaining buffer (handles last sentence without trailing whitespace)
        remaining = self._text_buffer.strip()
        if remaining:
            pcm = await self.synthesize(remaining)
            if pcm:
                await self._event_bus.emit_async(
                    EventType.TTS_AUDIO_CHUNK, {"audio": pcm}
                )
        self._text_buffer = ""
    finally:
        self._is_speaking = False
        await self._event_bus.emit_async(EventType.TTS_END)
```

### Pattern 3: 0x05 Binary Frame Encoding

**What:** TTS audio is already wired through the event system to `ConnectionManager.send_tts_audio()`, which packs a `0x05` prefix byte using `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)`. `TTSEngine` only needs to emit `EventType.TTS_AUDIO_CHUNK` with `{"audio": pcm_bytes}` — the framing is handled downstream.

**Verified from:** `smait/connection/protocol.py` line 17 (`TTS_AUDIO = 0x05`), `manager.py` lines 41-43 (subscription), lines 191-194 (`send_tts_audio`), lines 214-219 (`_on_tts_audio_chunk`).

```python
# Source: smait/connection/protocol.py (verified)
# TTSEngine emits:
self._event_bus.emit(EventType.TTS_AUDIO_CHUNK, {"audio": pcm_bytes})

# ConnectionManager handles automatically:
# _on_tts_audio_chunk -> send_tts_audio -> BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)
# = bytes([0x05]) + pcm_bytes
```

### Pattern 4: Mock KPipeline for Unit Tests

**What:** KPipeline is a synchronous callable that returns a synchronous generator. To mock it correctly, `mock_pipeline.return_value = iter([...])` — using `iter()` on a list, not a coroutine.

**Important:** The mock must return `KPipeline.Result`-like tuples. The simplest approach is to have the mock return tuples of `(graphemes_str, phonemes_str, numpy_float32_array)` which is what the backward-compat `__iter__` yields.

```python
# Source: existing tests/unit/test_tts.py (established pattern from Phase 1)
import numpy as np
from unittest.mock import MagicMock, patch

def make_mock_pipeline_cls(audio_chunks: list[np.ndarray]):
    """Create a mock KPipeline class that yields specified audio chunks."""
    mock_pipeline = MagicMock()
    # Each call to pipeline(text, ...) returns a generator of (gs, ps, audio) tuples
    mock_pipeline.return_value = iter(
        [(f"sentence_{i}", f"phonemes_{i}", chunk) for i, chunk in enumerate(audio_chunks)]
    )
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)
    return mock_pipeline_cls, mock_pipeline
```

**Critical trap:** Using `mock_pipeline.return_value = iter([...])` makes the mock only iterable ONCE. For tests that call `synthesize()` multiple times, recreate the mock or use `side_effect` with a factory.

### Pattern 5: Testing Streaming Interleave Order

**What:** The key Phase 2 test verifies that `speak_streaming` emits sentence 1 audio BEFORE synthesizing sentence 2, i.e., that streaming is truly incremental.

```python
# Approach: capture EventBus TTS_AUDIO_CHUNK calls in order
# and verify the first chunk is emitted before the second synthesis call

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

async def test_streaming_emits_first_sentence_before_second_synthesis(config, event_bus):
    """speak_streaming must emit sentence 1 audio before synthesizing sentence 2."""
    emission_order = []

    # Two calls to synthesize produce different audio
    call_count = 0
    async def mock_synthesize(text):
        nonlocal call_count
        call_count += 1
        emission_order.append(f"synthesize:{call_count}:{text[:10]}")
        return b'\x00' * 100

    original_emit = event_bus.emit_async
    async def tracked_emit(event_type, data=None):
        if event_type == EventType.TTS_AUDIO_CHUNK:
            emission_order.append("emit:audio_chunk")
        await original_emit(event_type, data)

    engine = TTSEngine(config, event_bus)
    engine.synthesize = mock_synthesize
    event_bus.emit_async = tracked_emit

    async def token_stream():
        for token in ["Hello world. ", "How are you?"]:
            yield token

    await engine.speak_streaming(token_stream())

    # First sentence synthesized, then emitted, then second sentence synthesized
    assert emission_order[0].startswith("synthesize:1"), "First synthesis must be first"
    assert emission_order[1] == "emit:audio_chunk", "First emit must follow first synthesis"
    assert emission_order[2].startswith("synthesize:2"), "Second synthesis must be after first emit"
```

### Anti-Patterns to Avoid

- **Using `emit()` instead of `emit_async()` in `speak_streaming()`:** `emit()` schedules async handlers as tasks but does not await them. `TTS_END` can fire before `TTS_AUDIO_CHUNK` handlers complete. Always use `emit_async()` in async contexts.
- **Testing with `asyncio.run()` when already in async context:** All test functions use `asyncio_mode = "auto"` — define them as `async def`, not sync with `asyncio.run()`.
- **Assuming `audio` from KPipeline is always numpy:** In the real model, `audio` is `torch.FloatTensor`. Check `hasattr(audio, 'numpy')` and call `.numpy()` before conversion, or use `np.array(audio)`.
- **Re-using a consumed `iter()` mock:** `iter(list)` is consumed after one iteration. Use `side_effect` with a factory for tests that call synthesize multiple times.
- **Calling `init_model()` in unit tests:** Downloads model weights from HuggingFace. Never call real `init_model()` in unit tests. Mock `kokoro.KPipeline` at the library boundary.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio chunking/phonemization | Custom text chunker | KPipeline internal chunker | KPipeline handles 510 phoneme limit internally; re-chunking outside creates duplicate splits |
| Binary frame framing | Custom frame builder in TTSEngine | `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm)` in ConnectionManager | Framing logic already exists and is tested; TTSEngine should remain protocol-agnostic |
| Async event ordering | Custom lock/semaphore | `emit_async()` | EventBus already provides awaited delivery semantics |
| Voice file download | Custom HF downloader | `KPipeline.load_voice()` (auto-download) | Called automatically on first `pipeline(text, voice=...)` call |

**Key insight:** Phase 2 is NOT about adding new infrastructure. The KPipeline, the event bus, the 0x05 frame protocol, and the test infrastructure all exist. The work is fixing the `emit` vs `emit_async` gap, the regex end-of-string flush, and adding streaming interleave tests.

---

## Common Pitfalls

### Pitfall 1: `emit()` vs `emit_async()` Race in `speak_streaming`

**What goes wrong:** `speak_streaming` calls `self._event_bus.emit(EventType.TTS_AUDIO_CHUNK, ...)` (sync `emit`). The `ConnectionManager._on_tts_audio_chunk` handler is an async coroutine — it gets scheduled as a task but may not execute before `TTS_END` is emitted. The Android device receives TTS_END control frame before the last audio chunk.

**Why it happens:** `EventBus.emit()` uses `loop.create_task()` for coroutine handlers. The task is scheduled but not awaited. The next line in `speak_streaming` continues immediately.

**How to avoid:** Use `await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, ...)` inside `speak_streaming` and `speak`. The `emit_async` method gathers all async handlers.

**Warning signs:** Android AudioTrack stops before all audio is played; TTS_END arrives before last audio frame in Wireshark capture.

### Pitfall 2: Last Sentence Dropped by Regex

**What goes wrong:** `SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')` requires whitespace after punctuation. The LLM's last token might be "." with no trailing space. The buffer contains "Hello." and the regex never matches. Only the explicit flush at the end of `speak_streaming` catches it — this is already present in the code, so this pitfall is mitigated. But for `_speak_by_sentence(text)`, if the text is "Hello. World" (no trailing space after "World"), the second sentence is captured correctly since `split()` is used.

**How to avoid:** The existing end-of-stream flush in `speak_streaming` handles this correctly. No regex fix needed. Document this behavior in tests.

**Warning signs:** Final sentence not synthesized — only manifests if the flush block has a bug.

### Pitfall 3: KPipeline Yields `torch.FloatTensor`, Not `numpy.ndarray`

**What goes wrong:** The installed `pipeline.py` shows `audio` is `result.output.audio` which is `KModel.Output.audio` — a `torch.FloatTensor`. The existing code in `tts.py` does `(audio * 32767).clip(-32768, 32767).astype(np.int16)` — numpy operations on a torch tensor work due to PyTorch's numpy bridge, but only for CPU tensors. On GPU (Phase 7), this silently fails or requires `.cpu().numpy()` first.

**How to avoid:** Add `audio = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)` before PCM conversion. This makes the code GPU-safe for Phase 7 without breaking Phase 2 CPU tests.

**Warning signs:** `RuntimeError: can't convert a given np.ndarray to a tensor - it doesn't have a numeric dtype` when running with real GPU model in Phase 7.

### Pitfall 4: `asyncio_mode = "auto"` Means No `asyncio.run()` in Tests

**What goes wrong:** Phase 1 tests used `asyncio.run(engine.synthesize(...))` in some places. With `asyncio_mode = "auto"` in `pyproject.toml`, pytest-asyncio manages the event loop. Calling `asyncio.run()` inside an already-running loop raises `RuntimeError: This event loop is already running`.

**How to avoid:** New streaming tests must be `async def` and use `await` directly, not `asyncio.run()`. The existing `test_tts.py` already has this right in some tests (`asyncio.run()` in sync test functions) — Phase 2 streaming tests should be `async def test_...` throughout.

**Warning signs:** `RuntimeError: This event loop is already running` in test output.

### Pitfall 5: Mock KPipeline Iterator Consumed After First Call

**What goes wrong:** `mock_pipeline.return_value = iter([(None, None, audio_chunk)])` — `iter()` on a list creates a one-shot iterator. If `synthesize()` is called twice (for two sentences), the second call gets an empty iterator and returns `b""`.

**How to avoid:** Use `side_effect` with a callable that creates a fresh iterator each time:
```python
mock_pipeline.side_effect = lambda text, **kwargs: iter([(None, None, audio_chunk)])
```

**Warning signs:** Second sentence returns empty bytes even though mock_pipeline was set up.

---

## Code Examples

Verified patterns from the installed source and existing codebase:

### KPipeline Backward-Compatible Iteration

```python
# Source: installed kokoro/pipeline.py v0.9.4, lines 338-349
# KPipeline.Result.__iter__ yields (graphemes, phonemes, audio)
# audio is torch.FloatTensor or None (if model=False)
for graphemes, phonemes, audio in self._pipeline(text, voice=self._voice, speed=1.0):
    if audio is None:
        continue  # Quiet pipeline (model=False), skip
    # audio: torch.FloatTensor shape [T] at 24kHz
    audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio)
    pcm = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    pcm_parts.append(pcm.tobytes())
```

### BinaryFrame.pack for 0x05 TTS Frames

```python
# Source: smait/connection/protocol.py (verified)
# TTSEngine emits — ConnectionManager packs automatically
self._event_bus.emit(EventType.TTS_AUDIO_CHUNK, {"audio": pcm_bytes})
# OR in async context:
await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm_bytes})

# ConnectionManager._on_tts_audio_chunk (already wired):
frame = BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)
# = bytes([0x05]) + pcm_bytes
await self._client.send(frame)
```

### Async Streaming Test Pattern

```python
# Source: analysis of smait/core/events.py + pyproject.toml asyncio_mode=auto
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from smait.output.tts import TTSEngine
from smait.core.events import EventType

@pytest.mark.asyncio  # Optional with asyncio_mode=auto
async def test_streaming_emits_before_full_synthesis(config, event_bus):
    audio_chunk = np.zeros(24000, dtype=np.float32)  # 1s silence

    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda text, **kwargs: iter(
        [(None, None, audio_chunk)]
    )
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

    emitted_chunks = []
    async def capture_audio(data):
        emitted_chunks.append(data)

    event_bus.subscribe(EventType.TTS_AUDIO_CHUNK, capture_audio)

    with patch("kokoro.KPipeline", mock_pipeline_cls):
        engine = TTSEngine(config, event_bus)
        await engine.init_model()

    async def token_stream():
        yield "Hello world. "
        yield "How are you?"

    await engine.speak_streaming(token_stream())

    assert len(emitted_chunks) == 2, f"Expected 2 audio chunks, got {len(emitted_chunks)}"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `emit()` in async streaming | `emit_async()` in async streaming | Phase 2 fix | Prevents TTS_END race with last audio chunk |
| `(audio * 32767).astype(np.int16)` | `.cpu().numpy()` before conversion | Phase 2 hardening | GPU-safe for Phase 7 |
| `asyncio.run()` in TTS tests | `async def` tests with `await` | Phase 2 | Correct pytest-asyncio usage |

**Already correct from Phase 1:**
- `KPipeline(lang_code='a')` — correct initialization
- Generator iteration `for gs, ps, audio in pipeline(...)` — correct API usage
- `TTSConfig.stream_by_sentence = True` — default streaming enabled
- `_voice = getattr(config.tts, 'voice', 'af_heart')` — voice config
- `FrameType.TTS_AUDIO = 0x05` + `ConnectionManager.send_tts_audio()` — protocol wiring complete

---

## Open Questions

1. **`torch.FloatTensor` vs `numpy.ndarray` in unit tests**
   - What we know: Real KPipeline yields `torch.FloatTensor` for `audio`; existing unit tests mock with `numpy.ndarray` which also works
   - What's unclear: Whether the CPU/GPU guard (`audio.cpu().numpy()`) should be added in Phase 2 or deferred to Phase 7 when GPU is available for testing
   - Recommendation: Add the guard in Phase 2 (`if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()`) — it costs nothing and prevents Phase 7 surprises. The unit test mock can use numpy since backward-compat `__iter__` just yields whatever was put in the mock.

2. **Sentence boundary for mid-word LLM tokens**
   - What we know: LLM may stream "Hello" + " world" + "." as three separate tokens, never emitting "." at end of a chunk
   - What's unclear: How often the buffering correctly accumulates these before detecting a boundary
   - Recommendation: No change needed — the buffer accumulation loop already handles this correctly; punctuation and following space will appear in subsequent tokens.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.0.0+ with pytest-asyncio 0.21.0+ |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) — `asyncio_mode = "auto"` |
| Quick run command | `pytest tests/unit/test_tts.py -x -q` |
| Full suite command | `pytest tests/ --cov=smait --cov-report=term-missing` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TTS-01 | `init_model()` instantiates `KPipeline(lang_code='a')` | unit (mock) | `pytest tests/unit/test_tts.py::test_correct_class_imported -x` | YES (passing from Phase 1) |
| TTS-01 | `synthesize()` iterates KPipeline generator, concatenates PCM | unit (mock) | `pytest tests/unit/test_tts.py::test_synthesize_uses_generator -x` | YES (passing from Phase 1) |
| TTS-01 | audio converted to PCM16 at 24kHz (float32 -> int16 clipped) | unit (mock) | `pytest tests/unit/test_tts.py::test_pcm_conversion_correct -x` | NO — Wave 0 |
| TTS-01 | `audio.cpu().numpy()` called when audio is torch tensor | unit (mock) | `pytest tests/unit/test_tts.py::test_torch_tensor_handled -x` | NO — Wave 0 |
| TTS-02 | `speak_streaming()` emits sentence 1 audio before sentence 2 synthesized | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_emits_interleaved -x` | NO — Wave 0 |
| TTS-02 | `speak_streaming()` flushes remaining buffer after stream ends | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_flushes_remainder -x` | NO — Wave 0 |
| TTS-02 | `speak_streaming()` emits TTS_START before any audio, TTS_END after all audio | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_event_order -x` | NO — Wave 0 |
| TTS-02 | `_speak_by_sentence()` splits on sentence boundaries correctly | unit | `pytest tests/unit/test_tts.py::test_sentence_splitting -x` | NO — Wave 0 |
| TTS-03 | `EventType.TTS_AUDIO_CHUNK` data is `bytes` | unit | `pytest tests/unit/test_tts.py::test_audio_chunk_is_bytes -x` | NO — Wave 0 |
| TTS-03 | `BinaryFrame.pack(FrameType.TTS_AUDIO, pcm)` produces `bytes([0x05]) + pcm` | unit | `pytest tests/unit/test_protocol.py::test_tts_audio_frame_encoding -x` | NO — Wave 0 |
| TTS-03 | `ConnectionManager._on_tts_audio_chunk` calls `send_tts_audio` with audio bytes | unit (mock websocket) | `pytest tests/unit/test_connection_manager.py::test_tts_audio_forwarded -x` | NO — Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/unit/test_tts.py -x -q`
- **Per wave merge:** `pytest tests/ --cov=smait --cov-report=term-missing`
- **Phase gate:** All TTS streaming tests green; `pytest tests/unit/test_tts.py -v` shows 0 failures

### Wave 0 Gaps

- [ ] `tests/unit/test_tts.py` — add streaming tests: `test_pcm_conversion_correct`, `test_torch_tensor_handled`, `test_streaming_emits_interleaved`, `test_streaming_flushes_remainder`, `test_streaming_event_order`, `test_sentence_splitting`, `test_audio_chunk_is_bytes`
- [ ] `tests/unit/test_protocol.py` — NEW file: `test_tts_audio_frame_encoding` (TTS-03 protocol verification)
- [ ] `tests/unit/test_connection_manager.py` — NEW file: `test_tts_audio_forwarded` (TTS-03 ConnectionManager integration)

---

## Sources

### Primary (HIGH confidence)

- Installed `venv/lib/python3.12/site-packages/kokoro/pipeline.py` (v0.9.4) — KPipeline `__call__` signature, `Result.__iter__` backward-compat, generator yield semantics, `split_pattern` behavior
- Installed `venv/lib/python3.12/site-packages/kokoro/__init__.py` — exports (`KPipeline`, `KModel`)
- `smait/connection/protocol.py` — `FrameType.TTS_AUDIO = 0x05`, `BinaryFrame.pack()` implementation
- `smait/connection/manager.py` — `send_tts_audio()`, `_on_tts_audio_chunk()` subscription, `TTS_START`/`TTS_END` handler wiring
- `smait/core/events.py` — `emit()` vs `emit_async()` semantics (task scheduling vs. gather)
- `smait/output/tts.py` — current implementation post-Phase 1: KPipeline integration, streaming buffer logic
- `tests/unit/test_tts.py` — existing test patterns (mock KPipeline, fixture usage)
- `smait/core/config.py` — `TTSConfig` (sample_rate=24000, stream_by_sentence=True)
- `.planning/phases/01-dependency-setup-stub-api-fixes/01-02-SUMMARY.md` — Phase 1 decisions affecting Phase 2

### Secondary (MEDIUM confidence)

- Phase 1 RESEARCH.md — established Kokoro API summary, voice defaults

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all API verified from installed source code (not just docs)
- Architecture: HIGH — existing `tts.py`, `events.py`, `protocol.py`, `manager.py` read directly; all integration points confirmed
- Pitfalls: HIGH — `emit` vs `emit_async` race traced through `events.py` source; torch/numpy issue confirmed from `pipeline.py`

**Research date:** 2026-03-09
**Valid until:** 2026-06-09 (Kokoro 0.9.4 API is stable; no known breaking changes planned)
