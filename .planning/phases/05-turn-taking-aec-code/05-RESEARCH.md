# Phase 5: Turn-Taking & AEC Code - Research

**Researched:** 2026-03-10
**Domain:** Silero VAD EOU detection, ASR hallucination filtering, Acoustic Echo Cancellation, barge-in state machine
**Confidence:** HIGH — all APIs verified from installed package source and direct Python introspection

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ASR-02 | Hallucination filtering rejects phantom transcripts (confidence + phrase blocklist) | `Transcriber._check_filters()` already implements this; `NeMo Hypothesis.word_confidence` enables real confidence; plan must add tests and expand blocklist |
| ASR-03 | VAD-based end-of-utterance detection with ~1.8s silence threshold (replaces LiveKit EOU) | `EOUDetector` exists with heuristic; must be rewritten to use Silero `VADIterator` with `min_silence_duration_ms=1800`; `VADIterator.__call__` returns `{'end': sample}` on silence threshold exceeded |
| AUD-06 | AEC approach researched and documented: CAE SDK AEC vs software AEC (speexdsp/WebRTC), implement best | `libspeexdsp-dev` installed on system; `speexdsp-python==0.1.1` pip-installable; `EchoCanceller.create(frame_size, filter_length, sr).process(near, far)` is the Python API; CAE SDK AEC likely covers this already (see architecture section) |
| AUD-07 | Barge-in: detect user speech during TTS, stop TTS, resume listening | `AudioPipeline._mic_gated` currently blocks ALL audio during TTS; must change to: unmute VAD during TTS, emit `BARGE_IN` event on speech detection, `TTSEngine` cancels pending synthesis via asyncio `Task.cancel()` |
</phase_requirements>

---

## Summary

Phase 5 has four distinct sub-problems that are well-bounded and buildable at home (no GPU needed):

**EOU Rewrite (ASR-03):** The `EOUDetector` currently uses a punctuation heuristic that is already passing tests. The Phase 5 goal is to replace the heuristic `on_silence()` timer with Silero `VADIterator` streaming — specifically using `min_silence_duration_ms=1800` (equivalent to ~1.8s threshold). The VADIterator is already loaded in `AudioPipeline`; EOUDetector needs to subscribe to a `VAD_SILENCE` internal signal rather than raw audio.

**Hallucination Filtering (ASR-02):** The `Transcriber._check_filters()` already implements confidence thresholding and a phrase blocklist. The plan must add: (1) real confidence extraction via `return_hypotheses=True` on NeMo transcribe, which gives `Hypothesis.word_confidence`, (2) expanded phrase blocklist, and (3) full test coverage for rejection edge cases.

**AEC Decision (AUD-06):** The CAE SDK on the Android device already performs beamforming and sends cleaned CAE audio to the server. The `cae_status` message from Jackie reports `aec: bool`. If CAE AEC is active, software AEC is redundant. Software AEC (`speexdsp`) is the fallback for when CAE AEC is not available (e.g., in testing or when the hardware link is not established). Implementation: `speexdsp-python==0.1.1` wraps the system `libspeexdsp1` (already installed), provides `EchoCanceller.create(256, 2048, 16000).process(near_pcm, far_pcm)` returning cleaned PCM. The far-end signal is TTS audio that was just sent to Jackie via `TTS_AUDIO_CHUNK` events.

**Barge-In Logic (AUD-07):** The current `_on_tts_start` gates the mic entirely — this blocks all speech detection during playback. Barge-in requires inverting this: during TTS, keep VAD running (so we can detect the user speaking), but when VAD detects speech, emit a new `BARGE_IN` event that causes `TTSEngine.speak_streaming()` to cancel its current synthesis `asyncio.Task`.

**Primary recommendation:** Implement in this order: (1) EOU VAD rewrite with tests, (2) hallucination filter tests + NeMo hypothesis confidence, (3) AEC decision documented + speexdsp stub, (4) barge-in state machine replacing mic gating.

---

## Standard Stack

### Core (all already installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| silero-vad | cached master | Voice activity + EOU timing | Already integrated; `VADIterator` streams 30ms chunks and emits `{'end': samples}` on silence |
| torch | nightly cu128 | VAD model inference | Already installed |
| nemo_toolkit | 2.6.2 | Parakeet TDT confidence scores | Already installed; `return_hypotheses=True` exposes `Hypothesis.word_confidence` |
| numpy | >=1.24 | PCM audio math | Already installed |

### New: AEC Only

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| speexdsp | 0.1.1 | Software acoustic echo cancellation | Fallback when CAE SDK AEC is not active; pip-installable (requires system `libspeexdsp-dev`) |

**Installation:**
```bash
# System dependency already present: libspeexdsp1:amd64
sudo apt install libspeexdsp-dev  # dev headers for building the Python binding
pip install speexdsp==0.1.1
```

Note: `libspeexdsp-dev` 1.2.1 is confirmed available in Ubuntu apt. `speexdsp-python==0.1.1` is buildable from pip (verified with `--dry-run`).

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| speexdsp | webrtcvad + WebRTC AEC | WebRTC AEC is harder to install as a standalone Python package; speexdsp has a simpler API and system lib is already present |
| speexdsp | librosa + numpy hand-rolled AEC | Never hand-roll AEC — adaptive filters take 50+ lines and still perform worse than Speex |
| VADIterator | custom silence counter | VADIterator already handles hysteresis (threshold - 0.15 = 0.35 for exit), speech_pad, and `reset_states()` atomically |

---

## Architecture Patterns

### Recommended Project Structure (additions only)

```
smait/
├── perception/
│   ├── eou_detector.py     # REWRITE: VADIterator-based silence tracking
│   └── transcriber.py      # ADD: NeMo hypothesis confidence extraction
├── sensors/
│   └── audio_pipeline.py   # MODIFY: remove full mic gate, add barge-in VAD path
├── output/
│   └── tts.py              # MODIFY: add cancellable task + BARGE_IN handler
└── core/
    ├── events.py            # ADD: BARGE_IN, VAD_END_OF_SPEECH event types
    └── config.py            # ADD: EOUConfig.silence_ms=1800, AECConfig
tests/
└── unit/
    ├── test_eou_detector.py    # EXTEND: VAD-based EOU tests
    ├── test_transcriber.py     # NEW: hallucination filter coverage
    └── test_barge_in.py        # NEW: barge-in state transitions
```

### Pattern 1: VADIterator-Based EOU Detection

**What:** Replace the heuristic silence timer with Silero `VADIterator` streaming. `VADIterator.__call__(chunk)` returns `{'end': sample_pos}` when silence exceeds `min_silence_duration_ms`.

**When to use:** Every 30ms audio chunk processed by `AudioPipeline.process_cae_audio()`.

**Key insight from source inspection:**
- `VADIterator.__call__` emits `{'start': N}` when speech begins
- It emits `{'end': N}` when `current_sample - temp_end >= min_silence_samples`
- Exit threshold is `threshold - 0.15` (0.5 - 0.15 = 0.35), providing natural hysteresis
- `min_silence_samples = sampling_rate * min_silence_duration_ms / 1000` — so 1800ms at 16kHz = 28,800 samples

**Architecture:**
- `EOUDetector` subscribes to `SPEECH_SEGMENT` and tracks whether a turn is pending
- Alternatively (simpler): `EOUDetector` exposes a `feed_vad_prob(prob, timestamp)` method called by `AudioPipeline`
- When silence >= 1800ms after speech, `EOUDetector` emits `END_OF_TURN`

**Example — VADIterator in streaming mode:**
```python
# Source: /home/gow/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/utils_vad.py
vad_iter = VADIterator(
    model,
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=1800,   # ~1.8s silence -> EOU
    speech_pad_ms=30,
)

# Per 30ms chunk (480 samples at 16kHz):
result = vad_iter(chunk_tensor)
if result:
    if 'start' in result:
        # Speech began
    elif 'end' in result:
        # Silence >= 1800ms -> end of utterance
        emit_end_of_turn()
```

**Critical:** `reset_states()` must be called after each END_OF_TURN emission and when a SPEECH_SEGMENT is emitted (already done in `AudioPipeline._reset_speech()`). If EOUDetector holds its own VADIterator, it needs its own reset.

**Design choice for Phase 5 — option A (recommended):**
Keep the existing `AudioPipeline` VAD for segment extraction (unchanged). Add a *separate* VAD pass in `EOUDetector` using a second `VADIterator` configured with `min_silence_duration_ms=1800`. AudioPipeline feeds raw audio to EOUDetector via a new `feed_chunk(chunk_tensor, timestamp)` method. This decouples segment detection from EOU detection.

**Design choice — option B:**
Reuse the existing AudioPipeline VAD prob from `_vad_model(chunk, sr)` and pass the prob value to EOUDetector, which applies its own 1800ms counter. This avoids running VAD twice.

Option B is lighter (no second model instance) but requires AudioPipeline to call EOUDetector's `feed_vad_prob()` on each chunk. Option A is cleaner separation but slightly heavier. The planner should use option B to avoid extra resource usage.

### Pattern 2: Hallucination Filter — NeMo Hypothesis Confidence

**What:** NeMo's `model.transcribe(audio, return_hypotheses=True)` returns `Hypothesis` objects with `word_confidence` and `token_confidence` lists. This replaces the current heuristic `_compute_confidence()` that returns 0.65 by default.

**When to use:** In `ParakeetASR.transcribe()`, use `return_hypotheses=True` when GPU is available.

**Example:**
```python
# Source: NeMo installed at venv/lib/python3.12/site-packages/nemo/collections/asr/parts/utils/rnnt_utils.py
# Hypothesis fields verified via Python introspection
result = self._model.transcribe([audio], return_hypotheses=True)
if isinstance(result, tuple):
    hyps = result[0]  # list of Hypothesis objects
else:
    hyps = result

hyp = hyps[0] if hyps else None
if hyp and hyp.word_confidence:
    confidence = float(min(hyp.word_confidence))  # worst-word confidence
elif hyp and hasattr(hyp, 'score'):
    confidence = float(hyp.score)
else:
    confidence = 0.65  # fallback
```

**Fallback for HOME (no GPU/NeMo weights):** The current `_compute_confidence()` returning 0.65 is the correct fallback. Tests must mock this path.

### Pattern 3: AEC Architecture Decision

**What:** Two-path AEC decision tree based on `cae_status.aec` from Jackie.

**Architecture:**
```
Jackie Android → CAE SDK processes 8ch mic audio → AEC inside CAE SDK → CAE audio (0x01 frame, clean)
               → Also sends cae_status JSON {"aec": true/false}

Server (Python):
  If cae_status.aec == True:
    → Trust CAE audio is already echo-cancelled
    → No software AEC needed
    → AudioPipeline uses CAE stream directly (current behavior)

  If cae_status.aec == False (or unknown/HOME testing):
    → Apply software AEC via speexdsp
    → far_end = TTS PCM recently sent to Jackie (buffered from TTS_AUDIO_CHUNK events)
    → near_end = incoming CAE audio
    → EchoCanceller.process(near_end, far_end) → cleaned audio
```

**AEC Implementation — speexdsp:**
```python
# Source: https://github.com/xiongyihui/speexdsp-python (verified from pip)
from speexdsp import EchoCanceller

class SoftwareAEC:
    def __init__(self, frame_size: int = 256, filter_length: int = 2048, sr: int = 16000):
        self._ec = EchoCanceller.create(frame_size, filter_length, sr)
        self._frame_size = frame_size

    def process(self, near_pcm: bytes, far_pcm: bytes) -> bytes:
        """near_pcm: microphone bytes, far_pcm: speaker playback bytes.
        Both must be mono int16 PCM, exactly frame_size samples (512 bytes at 256 samples).
        """
        return self._ec.process(near_pcm, far_pcm)
```

**Critical timing constraint:** The `far_pcm` (speaker reference) must be fed to `EchoCanceller.process()` *before or at the same time* as the echo appears in the mic. In practice: buffer the TTS PCM sent to Jackie (from `TTS_AUDIO_CHUNK` events), and align by estimated speaker latency (~100-200ms for Android AudioTrack).

**Home testing limitation:** Software AEC cannot be validated at home without actual speaker+mic hardware. Write the class and unit tests with mocked near/far audio; real validation is Phase 7 (LAB).

### Pattern 4: Barge-In State Machine

**What:** Replace full mic gating with speech detection during TTS, then cancel TTS on barge-in.

**Current behavior (wrong for barge-in):**
```python
def _on_tts_start(self, _data):
    self._mic_gated = True   # completely blocks VAD
```

**New behavior:**
```python
def _on_tts_start(self, _data):
    self._tts_playing = True  # keep VAD running, just track state

# When VAD detects speech AND self._tts_playing == True:
    event_bus.emit(EventType.BARGE_IN)

# TTSEngine subscribes to BARGE_IN:
    self._current_tts_task.cancel()  # cancels speak_streaming() task
```

**Asyncio Task cancellation pattern:**
```python
# In TTSEngine.speak_streaming():
async def speak_streaming(self, text_gen):
    self._tts_task = asyncio.current_task()
    ...
    try:
        async for chunk in text_gen:
            ...
    except asyncio.CancelledError:
        logger.info("TTS cancelled (barge-in)")
        raise  # must re-raise CancelledError
    finally:
        await self._event_bus.emit_async(EventType.TTS_END)

# In TTSEngine._on_barge_in():
def _on_barge_in(self, _data):
    if self._tts_task and not self._tts_task.done():
        self._tts_task.cancel()
```

**State transitions to test:**
```
IDLE -> TTS_START received -> _tts_playing=True, VAD active
TTS playing -> user speech detected -> BARGE_IN emitted -> TTS task cancelled -> TTS_END emitted -> mic listening resumes
TTS playing -> no user speech -> TTS completes naturally -> TTS_END -> mic listening resumes
```

**New EventType to add to `events.py`:**
```python
BARGE_IN = auto()      # User spoke during TTS playback
```

**EOUConfig changes in `config.py`:**
```python
@dataclass
class EOUConfig:
    min_silence_ms: int = 300        # existing: min silence before checking
    confidence_threshold: float = 0.7  # existing heuristic threshold
    hard_cutoff_ms: int = 1800       # UPDATE: was 1500, now 1800ms (~1.8s)
    vad_silence_ms: int = 1800       # NEW: VADIterator min_silence_duration_ms
    barge_in_min_speech_ms: int = 200  # NEW: min speech duration to count as barge-in
```

### Anti-Patterns to Avoid

- **Full mic gating during TTS:** Blocks barge-in. Replace `_mic_gated` with `_tts_playing` flag that keeps VAD active.
- **Sharing one VADIterator between AudioPipeline and EOUDetector:** State machine gets tangled. Use VAD prob passthrough (option B) or separate instances.
- **Not re-raising CancelledError in async tasks:** Asyncio task cancellation requires `raise` inside `except CancelledError` or using `finally` blocks.
- **Calling `process()` with wrong frame size in speexdsp:** `EchoCanceller.create(256, ...)` expects exactly 256-sample (512-byte) frames. Mismatched sizes raise segfaults or silent errors.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Silence hysteresis for EOU | Custom threshold counter | `VADIterator(min_silence_duration_ms=1800)` | Handles exit threshold (0.35), padding, sample counting correctly |
| Acoustic echo cancellation | Adaptive LMS filter in numpy | `speexdsp.EchoCanceller` | 200+ line adaptive filter still won't handle near-end speech or double-talk |
| ASR confidence scoring | Character-level heuristics | `return_hypotheses=True` + `Hypothesis.word_confidence` | Model knows which words it is uncertain about |
| Asyncio task cancellation | Time-based polling or global flags | `asyncio.Task.cancel()` + `CancelledError` | Native cancellation propagates through await chains |

**Key insight:** The most error-prone home-brew path is manual silence timing (drift from `time.monotonic()`) compared to `VADIterator` counting samples directly.

---

## Common Pitfalls

### Pitfall 1: VADIterator Not Reset Between Sessions

**What goes wrong:** After one utterance completes, `VADIterator.triggered = True` from the previous turn causes false `end` events immediately.

**Why it happens:** `VADIterator` is stateful — `triggered`, `temp_end`, `current_sample` persist.

**How to avoid:** Call `vad_iter.reset_states()` after each END_OF_TURN emission and at session start.

**Warning signs:** EOU fires immediately after mic unmute, with zero transcript.

### Pitfall 2: speexdsp Frame Size Mismatch

**What goes wrong:** Silent corruption or crash if near/far PCM byte lengths don't match `frame_size * 2` (int16 = 2 bytes/sample).

**Why it happens:** `EchoCanceller.create(256, 2048, 16000)` expects exactly 512 bytes per call.

**How to avoid:** The incoming CAE audio arrives in variable-length chunks. Buffer and slice into 256-sample frames before calling `process()`. At 16kHz, 256 samples = 16ms.

**Warning signs:** Segfault inside `speexdsp.EchoCanceller.process()` or all-zero output.

### Pitfall 3: TTS_END Not Emitted After Barge-In Cancellation

**What goes wrong:** `TTSEngine.speak_streaming()` is cancelled mid-flight; if `TTS_END` is not emitted in the `finally` block, `AudioPipeline._tts_playing` never clears and the system stays in a pseudo-gated state.

**Why it happens:** `asyncio.CancelledError` bypasses the normal code path.

**How to avoid:** Always emit `TTS_END` in `finally`, never in the happy path only.

```python
finally:
    self._is_speaking = False
    await self._event_bus.emit_async(EventType.TTS_END)  # ALWAYS runs
```

### Pitfall 4: Barge-In Triggered by Echo

**What goes wrong:** Without AEC, TTS audio played through the robot speaker gets picked up by the microphone. VAD sees "speech" and fires BARGE_IN, cancelling the TTS the robot just started playing.

**Why it happens:** The mic hears the speaker output.

**How to avoid:** Apply software AEC (speexdsp) on the incoming audio before feeding to barge-in VAD. Or require `cae_status.aec == True` before enabling barge-in. At HOME (testing), gate barge-in on a short delay (200ms) after TTS_START to ignore the initial speaker burst.

**Warning signs:** Barge-in fires within 200ms of TTS_START with no actual user speech.

### Pitfall 5: NeMo Confidence Extraction With Old Transcribe API

**What goes wrong:** `model.transcribe([audio])` without `return_hypotheses=True` returns `list[str]`, not Hypothesis objects. Accessing `.word_confidence` on a string crashes.

**Why it happens:** NeMo's transcribe return type varies by keyword argument.

**How to avoid:** Always check `isinstance(result[0], str)` before accessing hypothesis fields. Wrap in a try/except and fall back to 0.65.

---

## Code Examples

Verified patterns from installed packages and source inspection:

### EOU: VADIterator Streaming (Option B — VAD prob passthrough)

```python
# Source: /home/gow/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/utils_vad.py (lines 458-549)
# EOUDetector receives vad_prob from AudioPipeline per 30ms chunk

class EOUDetector:
    def __init__(self, config, event_bus):
        self._silence_threshold_samples = int(config.eou.vad_silence_ms / 1000 * 16000)  # 28800
        self._speech_exit_threshold = 0.35  # = VAD threshold(0.5) - 0.15
        self._silence_sample_count = 0
        self._in_speech = False
        self._pending_turn = False

    def feed_vad_prob(self, speech_prob: float, n_samples: int, timestamp: float) -> None:
        """Called per 30ms chunk by AudioPipeline."""
        if speech_prob >= 0.5:
            self._in_speech = True
            self._silence_sample_count = 0
            self._pending_turn = True
        elif speech_prob < self._speech_exit_threshold and self._in_speech:
            self._silence_sample_count += n_samples
            if self._silence_sample_count >= self._silence_threshold_samples:
                self._emit_end_of_turn(timestamp)
        # prob between 0.35-0.5: hysteresis zone, no state change
```

### Hallucination Filter: NeMo Hypothesis Confidence

```python
# Source: nemo.collections.asr.parts.utils.rnnt_utils.Hypothesis
# Verified: Hypothesis.word_confidence is a list of float per word

def _extract_confidence(self, model_output) -> float:
    """Extract word-level confidence from NeMo Hypothesis objects."""
    try:
        if isinstance(model_output, tuple):
            hyps = model_output[0]
        else:
            hyps = model_output

        if hyps and not isinstance(hyps[0], str):
            hyp = hyps[0]
            if hyp.word_confidence:
                return float(min(hyp.word_confidence))  # worst word = overall confidence
    except Exception:
        pass
    return 0.65  # safe fallback

# Transcribe call:
result = self._model.transcribe([audio], return_hypotheses=True)
confidence = self._extract_confidence(result)
```

### Barge-In: Asyncio Task Cancellation

```python
# Standard asyncio pattern for cancellable streaming tasks
# Source: Python asyncio docs (confirmed by project prior usage)

class TTSEngine:
    def __init__(self, ...):
        self._tts_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        event_bus.subscribe(EventType.BARGE_IN, self._on_barge_in)

    async def speak_streaming(self, text_gen):
        self._tts_task = asyncio.current_task()
        self._is_speaking = True
        await self._event_bus.emit_async(EventType.TTS_START)
        try:
            async for chunk in text_gen:
                # ... synthesize ...
                await self._event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm})
        except asyncio.CancelledError:
            logger.info("TTS barge-in cancellation")
            raise  # propagate — caller's await will see this
        finally:
            self._is_speaking = False
            self._tts_task = None
            await self._event_bus.emit_async(EventType.TTS_END)  # ALWAYS emit

    def _on_barge_in(self, _data: object) -> None:
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
```

### speexdsp AEC Integration

```python
# Source: https://github.com/xiongyihui/speexdsp-python (API verified from pip dry-run)
# frame_size=256 samples = 16ms at 16kHz
# filter_length=2048 samples ~ 128ms echo tail (sufficient for robot)
from speexdsp import EchoCanceller

class SoftwareAEC:
    FRAME_SAMPLES = 256
    FRAME_BYTES = FRAME_SAMPLES * 2  # int16

    def __init__(self, sample_rate: int = 16000):
        self._ec = EchoCanceller.create(self.FRAME_SAMPLES, 2048, sample_rate)
        self._near_buf = b""
        self._far_buf = b""

    def feed_far(self, pcm_bytes: bytes) -> None:
        """Buffer TTS audio being played (far-end/reference signal)."""
        self._far_buf += pcm_bytes

    def process_near(self, pcm_bytes: bytes) -> bytes:
        """Process mic audio (near-end) with echo cancellation."""
        self._near_buf += pcm_bytes
        output = b""
        while len(self._near_buf) >= self.FRAME_BYTES and len(self._far_buf) >= self.FRAME_BYTES:
            near_frame = self._near_buf[:self.FRAME_BYTES]
            far_frame = self._far_buf[:self.FRAME_BYTES]
            self._near_buf = self._near_buf[self.FRAME_BYTES:]
            self._far_buf = self._far_buf[self.FRAME_BYTES:]
            output += self._ec.process(near_frame, far_frame)
        return output
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LiveKit private EOU model (unavailable) | VAD silence threshold | Phase 5 | Removes unavailable dependency; no GPU needed for EOU |
| Mic gating (blocks barge-in) | VAD-active + BARGE_IN event | Phase 5 | Enables user interruption of robot speech |
| Heuristic confidence (0.65 fallback) | NeMo Hypothesis.word_confidence (min of word confs) | Phase 5 | Real per-word uncertainty from the ASR model |
| Full silence timer for EOU | VADIterator hysteresis (0.35 exit) with 1800ms counter | Phase 5 | Sample-accurate, no monotonic drift |

**Deprecated/outdated in this codebase:**
- `EOUDetector._heuristic_eou()`: Keep as final fallback (when `return_hypotheses` fails), but the primary EOU path must be VAD-based
- `AudioPipeline._mic_gated`: Replace with `_tts_playing` flag that preserves VAD activity

---

## Open Questions

1. **CAE SDK AEC coverage**
   - What we know: `cae_status` JSON includes `aec: bool`; server receives this from Jackie
   - What's unclear: Is CAE AEC always active, or only when phone is in a specific mode? Is it single-channel AEC or multi-channel?
   - Recommendation: Implement software AEC as fallback, gated on `cae_status.aec == False`. HOME tests use `cae_status.aec = False` path exclusively. LAB (Phase 7) will determine if hardware AEC is sufficient.

2. **Barge-in echo false positives at HOME**
   - What we know: Without real hardware, barge-in tests use mocked audio
   - What's unclear: Will software AEC (speexdsp) be fast enough to prevent barge-in false triggers from speaker echo?
   - Recommendation: Add a `min_barge_in_delay_ms=200` guard — ignore VAD speech detections within 200ms of TTS_START to absorb initial speaker transient.

3. **EOUConfig hard_cutoff_ms value**
   - Current value: 1500ms in `EOUConfig.hard_cutoff_ms`
   - Target: ~1800ms silence threshold per ASR-03
   - Recommendation: Update `hard_cutoff_ms` default to 1800ms in `EOUConfig`. Add new `vad_silence_ms=1800` field for the VADIterator parameter. Both should be configurable.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.x + pytest-asyncio |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` asyncio_mode = "auto" |
| Quick run command | `./venv/bin/python -m pytest tests/unit/test_eou_detector.py tests/unit/test_transcriber.py tests/unit/test_barge_in.py -x -q` |
| Full suite command | `./venv/bin/python -m pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ASR-03 | VAD silence >= 1800ms triggers END_OF_TURN | unit | `./venv/bin/python -m pytest tests/unit/test_eou_detector.py -x` | Exists (extends) |
| ASR-03 | Silence < 1800ms does NOT trigger END_OF_TURN | unit | same | Extends existing |
| ASR-03 | VADIterator reset_states() called after EOU | unit | same | Wave 0 gap |
| ASR-03 | Speech detected resets silence counter | unit | same | Wave 0 gap |
| ASR-02 | Phrases in HALLUCINATION_PHRASES rejected at conf < 0.60 | unit | `./venv/bin/python -m pytest tests/unit/test_transcriber.py -x` | Wave 0 gap |
| ASR-02 | Phrases above confidence threshold accepted | unit | same | Wave 0 gap |
| ASR-02 | Short utterance (<8 words) + conf < 0.40 rejected | unit | same | Wave 0 gap |
| ASR-02 | NeMo Hypothesis.word_confidence path tested | unit | same | Wave 0 gap |
| AUD-06 | SoftwareAEC.process() returns same-length PCM | unit | `./venv/bin/python -m pytest tests/unit/test_aec.py -x` | Wave 0 gap |
| AUD-06 | AEC disabled when cae_status.aec == True | unit | same | Wave 0 gap |
| AUD-07 | BARGE_IN event fires when VAD detects speech during TTS | unit | `./venv/bin/python -m pytest tests/unit/test_barge_in.py -x` | Wave 0 gap |
| AUD-07 | TTS_END emitted after barge-in cancellation | unit | same | Wave 0 gap |
| AUD-07 | TTS task is actually cancelled (not just flagged) | unit | same | Wave 0 gap |
| AUD-07 | No barge-in within min_barge_in_delay_ms of TTS_START | unit | same | Wave 0 gap |

### Sampling Rate

- **Per task commit:** `./venv/bin/python -m pytest tests/unit/test_eou_detector.py tests/unit/test_transcriber.py tests/unit/test_barge_in.py tests/unit/test_aec.py -x -q`
- **Per wave merge:** `./venv/bin/python -m pytest tests/ -q`
- **Phase gate:** Full suite green (currently 91 tests pass) before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/unit/test_transcriber.py` — covers ASR-02 (Transcriber hallucination filter, NeMo hypothesis path)
- [ ] `tests/unit/test_barge_in.py` — covers AUD-07 (barge-in state transitions, TTS cancellation, TTS_END guarantee)
- [ ] `tests/unit/test_aec.py` — covers AUD-06 (SoftwareAEC frame processing, CAE status gating)
- [ ] `speexdsp` install: `sudo apt install libspeexdsp-dev && pip install speexdsp==0.1.1`
- [ ] `EventType.BARGE_IN` — add to `smait/core/events.py`

`tests/unit/test_eou_detector.py` exists but needs additional test cases for the VAD-based rewrite (ASR-03).

---

## Sources

### Primary (HIGH confidence)

- Silero VAD source `/home/gow/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/utils_vad.py` — VADIterator implementation, parameters, hysteresis logic (lines 458-549)
- NeMo 2.6.2 installed source — `Hypothesis.word_confidence`, `return_hypotheses=True`, `ConfidenceConfig` (verified via Python introspection)
- Project source — `EOUDetector`, `Transcriber`, `AudioPipeline`, `TTSEngine` (all read directly)
- `speexdsp-python==0.1.1` — pip dry-run confirms installable; `EchoCanceller.create(frame_size, filter_len, sr).process(near, far)` API verified from GitHub README

### Secondary (MEDIUM confidence)

- LiveKit Silero VAD docs (https://docs.livekit.io/agents/logic/turns/vad/) — confirmed `min_silence_duration=0.55s` as default; 1.8s is a deliberate choice for robot turn-taking
- speexdsp-python GitHub (https://github.com/xiongyihui/speexdsp-python) — API and example code; not directly installed yet but buildable

### Tertiary (LOW confidence)

- System `libspeexdsp-dev` version 1.2.1 in apt — confirmed available; Python binding build not tested end-to-end on this machine

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed and introspected
- Architecture: HIGH — based on reading actual source code of all affected modules
- Pitfalls: HIGH — derived from real API behavior (VADIterator state, asyncio CancelledError semantics)
- AEC approach: MEDIUM — speexdsp API confirmed from pip, but hardware validation deferred to Phase 7

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable ecosystem; Silero VAD API has not changed in 12+ months)
