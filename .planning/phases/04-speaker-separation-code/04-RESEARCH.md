# Phase 4: Speaker Separation Code - Research

**Researched:** 2026-03-10
**Domain:** Dolphin AV-TSE integration, Silero VAD, audio-visual temporal sync, DOA disambiguation
**Confidence:** HIGH — vendored look2hear source is in the project; all APIs verified from actual source code

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SEP-01 | Dolphin AV-TSE loaded with correct API (`from look2hear.models import Dolphin`) | `look2hear/models/__init__.py` exports `Dolphin`; `Dolphin.from_pretrained("JusperLee/Dolphin")` confirmed in `dolphin.py` |
| SEP-02 | Audio input preprocessed to mono `[1, samples]` at 16kHz for Dolphin | `Dolphin.forward(input, mouth)` takes `input` shaped `[batch, samples]` — verified from dolphin.py line 1361 |
| SEP-03 | Lip frames preprocessed to 88x88 grayscale at 25fps for Dolphin | `VideoEncoder` inside Dolphin expects `[batch, 1, T, 88, 88, 1]` — current DolphinSeparator already does this correctly |
| SEP-04 | Audio-visual temporal sync via server-side monotonic timestamps | `LipExtractor.get_lip_frames(track_id, start_time, end_time)` + `SpeechSegment.start_time/end_time` both use `time.monotonic()` |
| SEP-05 | DOA angles integrated into engagement detector for multi-speaker disambiguation | `EngagementDetector._on_doa_update` already subscribes to `DOA_UPDATE`; `_select_primary_user` applies 1.2x score bonus — needs more precise angle-to-face mapping |
| SEP-06 | Fallback to CAE passthrough audio when Dolphin unavailable | `DolphinSeparator._passthrough()` already implemented; triggered when `_available=False` or model call fails |
| AUD-05 | Silero VAD segments speech from CAE audio with ring buffer alignment | `AudioPipeline` already has `process_cae_audio`, `process_raw_audio`, `RawAudioBuffer.extract()` — needs VAD silence threshold tuning |
</phase_requirements>

---

## Summary

Phase 4 is primarily a **correctness and test-coverage phase**, not a from-scratch implementation phase. The core `DolphinSeparator`, `AudioPipeline`, and `EngagementDetector` classes already exist with substantially correct implementations from prior phases. The existing tests (6 passing) verify the API imports and tensor shapes, but several behaviors have zero test coverage.

The key work is: (1) writing tests that drive out edge cases and verify behavioral contracts, (2) correcting the audio routing in `main.py` (currently passes `raw_audio` to Dolphin rather than `cae_audio`), (3) improving DOA integration in `EngagementDetector` to map angles to face positions, and (4) verifying the VAD ring buffer produces correctly sized segments.

**Primary recommendation:** Fix the audio routing bug first (raw audio should not be passed to Dolphin — CAE mono is the input), then write tests to lock in all SEP/AUD-05 behaviors. The stub code structure is mostly right; the planner should focus on incremental TDD tasks that add missing tests and fix the routing logic.

---

## Standard Stack

### Core (already vendored / installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| look2hear (vendored) | HEAD | Dolphin AV-TSE model | Only open AV-TSE with pretrained weights; vendored into `look2hear/` in project root |
| silero-vad | 6.2.1 | Voice activity detection | Sub-millisecond per chunk on CPU, already integrated |
| torch | nightly cu128 | GPU tensor operations | Required for sm_120 Blackwell; stable builds lack sm_120 kernels |
| numpy | >=1.24 | Audio/image array manipulation | Universal scientific Python standard |
| opencv-python | >=4.8 | Grayscale conversion, image resize | Already used throughout; fastest for uint8 image ops |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| huggingface_hub | >=0.20 | `Dolphin.from_pretrained()` weight download | Required by `PyTorchModelHubMixin`; already in requirements |
| safetensors | >=0.4 | Load `model.safetensors` from HF Hub | Required by `Dolphin.from_pretrained()` — `dolphin.py` line 1337 |
| vector_quantize_pytorch | any | ResidualVQ inside Dolphin | Already installed as part of look2hear requirements |
| beartype | any | Type checking in video_compoent.py | Already installed; emits deprecation warnings (harmless) |

**No new libraries needed for Phase 4.** All dependencies are already installed.

---

## Architecture Patterns

### Recommended Project Structure

```
smait/
├── perception/
│   ├── dolphin_separator.py   # DolphinSeparator — ALREADY EXISTS, minor fix needed
│   └── engagement.py          # EngagementDetector — DOA integration improvement needed
├── sensors/
│   └── audio_pipeline.py      # AudioPipeline + RawAudioBuffer — ALREADY EXISTS
└── main.py                    # HRISystem._wire_events — audio routing bug to fix

tests/unit/
├── test_dolphin_separator.py  # 6 tests GREEN — needs ~6 more for sync, fallback, edge cases
├── test_engagement.py         # 6 tests GREEN — needs DOA integration test
└── test_audio_pipeline.py     # NEW — verify VAD segmentation, ring buffer, silence threshold
```

### Pattern 1: Dolphin Forward Call (Verified from Source)

**What:** `Dolphin.forward(input, mouth)` takes exactly two positional arguments.
**Source:** `look2hear/models/dolphin.py` line 1361.

```python
# Source: look2hear/models/dolphin.py line 1361
def forward(self, input, mouth):
    mouth = self.video_encoder(mouth).permute(0, 2, 1).contiguous()
    # ... processes audio with video guidance
    return audio.unsqueeze(dim=1)  # Returns [batch, 1, samples]
```

**Critical:** The output is `[batch, 1, samples]`, not `[batch, samples]`. The current `_run_dolphin` code does `separated.squeeze().cpu().numpy()` — this works because squeeze removes the size-1 dimensions to give a 1D array, but the test at line 82 passes `fake_output = torch.zeros(1, 16000)` which is 2D not 3D. The real model returns `[1, 1, samples]` — squeeze still works correctly, but tests with fake outputs need to match the real shape.

### Pattern 2: Correct Audio Routing (BUG IN main.py)

**What goes wrong:** `HRISystem.on_speech_segment` at lines 221-223 passes `raw_audio` (4-channel interleaved) to Dolphin when available, using it as the primary source. This is wrong — Dolphin takes mono only. The CAE audio IS the correct mono input.

**Current (WRONG) logic in main.py lines 221-223:**
```python
# WRONG — passes 4-channel raw to Dolphin
channels = self._config.audio.channels_raw if segment.raw_audio is not None else 1
audio = segment.raw_audio if segment.raw_audio is not None else segment.cae_audio
separation = await self.dolphin_separator.separate(audio, lip_frames, channels)
```

**Correct logic:**
```python
# CORRECT — always use CAE mono for Dolphin; raw_audio not used for Dolphin
separation = await self.dolphin_separator.separate(
    segment.cae_audio,  # Always CAE mono (16kHz int16)
    lip_frames,
    channels=1,          # Always mono
)
```

**Why:** Dolphin's visual cues replace the need for multi-channel spatial audio. The CAE already provides beamformed directional audio. Mixing down 4-channel input inside `_run_dolphin` then discarding spatial info wastes computation and reduces quality vs. just using the CAE-processed mono stream.

### Pattern 3: Audio-Visual Temporal Sync

**What:** Match lip frames from `LipExtractor` buffer to a `SpeechSegment` using monotonic timestamps.
**When to use:** On every `SPEECH_SEGMENT` event.

```python
# Source: smait/main.py lines 213-218 — current implementation
lip_frames = []
if self.lip_extractor and self.session.target_track_id is not None:
    lip_frames = self.lip_extractor.get_lip_frames(
        self.session.target_track_id,
        segment.start_time,   # SpeechSegment.start_time (monotonic)
        segment.end_time,     # SpeechSegment.end_time (monotonic)
    )
```

**Sync invariant:** Both `LipROI.timestamp` (set in `lip_extractor.extract()`) and `SpeechSegment.start_time/end_time` (set in `audio_pipeline._emit_segment()`) use `time.monotonic()` on the server. This guarantees they share the same clock.

**Edge case:** If `lip_frames` is empty (face not visible during speech), `_run_dolphin` sets `video_tensor = None` and calls `model(audio_tensor)` audio-only. But per ARCHITECTURE.md: "Dolphin requires both audio and video tensors. No audio-only mode verified." The current code should skip Dolphin entirely and use passthrough when no lip frames exist.

### Pattern 4: DOA-to-Face Alignment (SEP-05)

**Current state:** `_select_primary_user` applies a flat 1.2x score bonus to ALL candidates when any DOA angle is present. It does NOT map the DOA angle to a specific face position.

**Correct approach:** Use face bounding box center X coordinate to estimate face angle relative to camera, then compute angular distance to DOA direction.

```python
def _doa_alignment_score(self, face_center_x: int, frame_width: int,
                          doa_angle: int, camera_fov_deg: float = 60.0) -> float:
    """Score face-to-DOA alignment. Returns 1.0 (best) to 0.5 (worst)."""
    # Map pixel X to camera angle: center=0, left=negative, right=positive
    normalized = (face_center_x / frame_width) - 0.5  # [-0.5, 0.5]
    face_angle_deg = normalized * camera_fov_deg
    # DOA angle: 0=front, -90=left, +90=right (matches camera convention)
    angular_distance = abs(face_angle_deg - doa_angle)
    # Score: 1.0 at 0 deg difference, 0.5 at 45 deg difference
    return max(0.5, 1.0 - angular_distance / 90.0)
```

**Note:** `FaceTrack` has `face_area` and `bbox` attributes — verify bbox contains center X. If not, the planner needs to add it.

### Pattern 5: VAD Silence Threshold (AUD-05)

**Current state in `AudioPipeline.process_cae_audio`:** Silence threshold uses `config.audio.min_speech_duration_ms` (250ms default) as the silence cutoff. This is actually checking `silence_ms >= min_speech_duration_ms` which is the MINIMUM SEGMENT DURATION check, not a silence end-of-speech threshold.

**Recommended configuration:**

| Parameter | Config Field | Value | Rationale |
|-----------|-------------|-------|-----------|
| VAD threshold | `audio.vad_threshold` | 0.5 | Default, works for most environments |
| Min silence for EOU | `audio.min_speech_duration_ms` | 300ms | Conservative — avoids cutting mid-sentence |
| Min segment to send | `MIN_SEGMENT_DURATION_S` | 0.5s | Already hardcoded, prevents Parakeet hallucinations |
| Ring buffer size | `audio.raw_buffer_seconds` | 30.0s | Enough for very long utterances |

### Anti-Patterns to Avoid

- **Passing 4-channel raw audio to Dolphin:** Dolphin is mono-only. The `used_multichannel` flag in `SeparationResult` is misleading — it should be removed or renamed.
- **Calling `model(audio_tensor)` without video:** Dolphin's `forward(self, input, mouth)` requires both args. No `mouth=None` default exists. Calling audio-only causes `TypeError`. Use passthrough instead when no lip frames.
- **Using `torch.no_grad()` instead of `torch.inference_mode()`:** `inference_mode` is faster and disables more autograd overhead. Update `_run_dolphin` for the real model call.
- **Assuming `Dolphin.forward` returns `[batch, samples]`:** It returns `audio.unsqueeze(dim=1)` = `[batch, 1, samples]`. The `.squeeze()` call handles this but tests must match.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Speaker model loading | Custom checkpoint loader | `Dolphin.from_pretrained("JusperLee/Dolphin")` | Already handles config.json + safetensors download from HF |
| Audio normalization | Custom int16→float32 scaler | Standard `/32768.0` division (already in code) | Correct for int16 PCM range |
| Ring buffer timestamps | Binary search or sorted dict | Linear scan in `_time_to_offset` (already written) | Buffer is small (30s); O(n) is fine |
| VAD silence detection | Custom energy-based VAD | Silero VAD (already integrated) | Enterprise-grade, sub-ms per chunk |
| Face-DOA angle mapping | Computer vision epipolar geometry | Camera FOV approximation (angular heuristic) | Sufficient precision for multi-speaker disambiguation |

---

## Common Pitfalls

### Pitfall 1: Audio Routing Bug — Raw Audio Sent to Dolphin

**What goes wrong:** `main.py` passes `segment.raw_audio` (4-channel interleaved int16) to `DolphinSeparator.separate()` when available, with `channels=4`. The internal mixdown in `_run_dolphin` averages channels, but this ignores that CAE beamforming has already done better spatial filtering.

**Why it happens:** The intent was to give Dolphin more raw signal — but Dolphin already has audio-visual guidance from lips; it doesn't benefit from multi-channel spatial info the way a beamformer does.

**How to avoid:** Always pass `segment.cae_audio` (mono 16kHz) with `channels=1` to `DolphinSeparator.separate()`. The raw 4-channel buffer exists only for potential future multi-channel Dolphin support.

**Warning signs:** `SeparationResult.used_multichannel=True` in production logs.

### Pitfall 2: Audio-Only Dolphin Call Crashes

**What goes wrong:** When `lip_frames` is empty, current `_run_dolphin` code sets `video_tensor = None` and calls `self._model(audio_tensor)`. The real `Dolphin.forward(self, input, mouth)` requires `mouth` — no default. Calling with one arg raises `TypeError: forward() missing 1 required positional argument: 'mouth'`.

**How to avoid:** In `separate()`, check `if not lip_frames: return self._passthrough(audio, channels, start)` BEFORE calling `_run_dolphin`. Never call `_run_dolphin` with empty lip_frames.

**Warning signs:** `TypeError` in logs during speech with face not visible.

### Pitfall 3: Dolphin Output Shape Mismatch in Tests

**What goes wrong:** Unit tests mock `model.return_value = torch.zeros(1, 16000)` (2D). Real model returns `audio.unsqueeze(dim=1)` = `[1, 1, 16000]` (3D). The `.squeeze()` still works for both (both give 1D output), but tests checking shape explicitly may mislead.

**How to avoid:** When writing new tests, use `torch.zeros(1, 1, 16000)` as the mock return value to match the real model's `unsqueeze(dim=1)` output.

### Pitfall 4: VAD Reset Called with No Model

**What goes wrong:** `AudioPipeline._reset_speech()` calls `self._vad_model.reset_states()` unconditionally. If model hasn't been loaded (e.g., in tests without `init_model()` called), this raises `AttributeError: 'NoneType' object has no attribute 'reset_states'`.

**How to avoid:** Guard: `if self._vad_model is not None: self._vad_model.reset_states()`.

**Note:** This bug exists in the current code and will surface in `test_audio_pipeline.py` tests.

### Pitfall 5: LipExtractor Buffer Produces 0 Frames for Long Speech

**What goes wrong:** `LIP_BUFFER_SECONDS = 5.0` at 15fps = 75 frames max. For a 10-second utterance, frames from the start of speech will have been evicted from the deque before the `SpeechSegment` arrives. `get_lip_frames(start_time, end_time)` returns 0 frames for the early window.

**How to avoid:** Use `get_recent_frames(track_id, count)` as a fallback when `get_lip_frames()` returns empty. The recent frames are better than nothing: they show current mouth state for a speaker who has been talking.

### Pitfall 6: DOA Angle Convention Mismatch

**What goes wrong:** CAE SDK reports DOA angles in degrees (0-360 or -180 to +180 depending on firmware). Camera field of view convention may differ. Applying DOA as-is to face position scoring gives wrong results (penalizes correct face, rewards wrong one).

**How to avoid:** Log raw DOA values during testing. Define the convention explicitly in code comments. Initial implementation should use a generous tolerance (±30 degrees) until convention is confirmed from hardware.

---

## Code Examples

### Correct Dolphin Separation Call

```python
# Source: look2hear/models/dolphin.py line 1361 — verified forward signature
async def _run_dolphin(self, audio, lip_frames, channels, start):
    # Always mono — CAE audio is passed in at channels=1
    audio_float = audio.astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).to(self._device)  # [1, samples]

    # Build video tensor — MUST have lip_frames (no audio-only mode)
    grayscale_frames = []
    for roi in lip_frames:
        gray = cv2.cvtColor(roi.image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (88, 88), interpolation=cv2.INTER_LINEAR)
        grayscale_frames.append(gray)

    lip_stack = np.stack(grayscale_frames, axis=0)   # [T, 88, 88]
    lip_stack = lip_stack[..., np.newaxis]             # [T, 88, 88, 1]
    video_tensor = torch.from_numpy(
        lip_stack[np.newaxis, np.newaxis]              # [1, 1, T, 88, 88, 1]
    ).float().to(self._device)

    # Use inference_mode (faster than no_grad)
    with torch.inference_mode():
        output = self._model(audio_tensor, video_tensor)  # [1, 1, samples]

    separated_np = output.squeeze().cpu().numpy()  # [samples] 1D float32
    # ...
```

### Correct separate() Entry Point (with early exit for no lip_frames)

```python
# Guard: no lip_frames → passthrough (Dolphin requires video tensor)
async def separate(self, audio, lip_frames, channels=1):
    start = time.monotonic()

    if not self._available or self._model is None:
        return self._passthrough(audio, channels, start)

    if not lip_frames:
        # Cannot call Dolphin without visual input — use CAE passthrough
        logger.debug("No lip frames for target speaker — using CAE passthrough")
        return self._passthrough(audio, channels, start)

    try:
        return await self._run_dolphin(audio, lip_frames, channels, start)
    except Exception:
        logger.exception("Dolphin separation failed, using passthrough")
        return self._passthrough(audio, channels, start)
```

### Corrected Audio Routing in main.py

```python
# Source: smait/main.py on_speech_segment handler — fix required
async def on_speech_segment(segment):
    lip_frames = []
    if self.lip_extractor and self.session.target_track_id is not None:
        lip_frames = self.lip_extractor.get_lip_frames(
            self.session.target_track_id,
            segment.start_time,
            segment.end_time,
        )
        # Fallback: if segment window missed buffered frames, get recent ones
        if not lip_frames:
            lip_frames = self.lip_extractor.get_recent_frames(
                self.session.target_track_id, count=25
            )

    # ALWAYS use CAE mono audio for Dolphin — not raw 4-channel
    separation = await self.dolphin_separator.separate(
        segment.cae_audio,   # CAE-beamformed mono 16kHz int16
        lip_frames,
        channels=1,
    )
```

### VAD Reset Guard

```python
# Source: smait/sensors/audio_pipeline.py _reset_speech — guard needed
def _reset_speech(self):
    self._in_speech = False
    self._speech_buffer = []
    self._speech_start_time = None
    self._silence_start_time = None
    if self._vad_model is not None:    # Guard against pre-init calls
        self._vad_model.reset_states()
```

### DOA Alignment Score (SEP-05)

```python
# To add in smait/perception/engagement.py _select_primary_user
def _doa_score_for_face(self, track: FaceTrack) -> float:
    """Return DOA alignment multiplier for a face track (1.0 = perfect, 0.5 = opposed)."""
    if self._last_doa_angle is None:
        return 1.0  # No DOA data — no penalty
    if not hasattr(track, 'bbox') or track.bbox is None:
        return 1.0  # No bbox — no penalty
    x1, _, x2, _ = track.bbox
    face_center_x = (x1 + x2) / 2
    # Assume 640px wide frame, 60-degree FOV
    frame_width = 640
    fov = 60.0
    normalized = (face_center_x / frame_width) - 0.5
    face_angle_deg = normalized * fov
    angular_distance = abs(face_angle_deg - self._last_doa_angle)
    return max(0.5, 1.0 - angular_distance / 90.0)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Stub `from dolphin import DolphinModel` | `from look2hear.models import Dolphin` | Phase 1 | Fixed ImportError |
| Audio shape `[batch, channels, samples]` | Mono `[batch, samples]` | Phase 1 | Correct tensor |
| Video shape `[batch, T, H, W, C]` | `[batch, 1, T, 88, 88, 1]` | Phase 1 | Correct tensor |
| Pass raw 4-ch to Dolphin | Pass CAE mono only | Phase 4 (this phase) | Eliminates unnecessary mixdown |
| No lip_frames → audio-only mode | No lip_frames → passthrough | Phase 4 (this phase) | Prevents TypeError crash |
| DOA: flat 1.2x bonus for all faces | DOA: per-face angle proximity score | Phase 4 (this phase) | Actual disambiguation |

**Deprecated/outdated:**
- `SeparationResult.used_multichannel`: misleading field — will always be False after audio routing fix; remove or repurpose.
- `_run_dolphin(audio, lip_frames, channels, start)` `channels` parameter: no longer needed when always called with `channels=1`; keep for now to avoid API churn.

---

## Open Questions

1. **FaceTrack.bbox availability**
   - What we know: `FaceTrack` has `face_area` and `landmarks` — bbox may not be directly exposed
   - What's unclear: Whether bbox (x1,y1,x2,y2) is available for face center X calculation
   - Recommendation: Check `smait/perception/face_tracker.py` FaceTrack dataclass during planning; add bbox field if absent (MediaPipe provides it)

2. **Dolphin minimum viable frame count**
   - What we know: Real-time speech segments can be 0.5–5 seconds; at 15fps that's 7–75 lip frames
   - What's unclear: Whether Dolphin has a minimum T (frames) requirement for the VideoEncoder
   - Recommendation: Test with T=5 (minimum speech segment = 0.5s at 10fps). If model crashes for small T, pad with zeros.

3. **DOA angle convention from CAE SDK**
   - What we know: `DOA_UPDATE` event carries `{"angle": int}` — exact range and reference direction TBD
   - What's unclear: Is angle 0=front? 0=north? Is range 0-360 or -180 to +180?
   - Recommendation: Log raw DOA values in first integration test. Hardcode assumption with TODO until hardware confirmed. Keep DOA score optional (returns 1.0 when convention unknown).

4. **Dolphin `from_pretrained` download behavior in HOME environment**
   - What we know: `Dolphin.from_pretrained("JusperLee/Dolphin")` downloads `model.safetensors` from HuggingFace Hub
   - What's unclear: Whether weights have been pre-downloaded in this environment (Phase 4 is HOME — no GPU needed for tests with mocks)
   - Recommendation: Phase 4 tests use mocked Dolphin model. Weight download deferred to Phase 7 (LAB). The `test_look2hear_importable` test already passes (module importable, no weights needed).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | `pyproject.toml` (rootdir: `/home/gow/.openclaw/workspace/projects/SMAIT-v3`) |
| Quick run command | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py tests/unit/test_engagement.py -x -q` |
| Full suite command | `./venv/bin/python -m pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SEP-01 | `from look2hear.models import Dolphin` succeeds | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_look2hear_importable -x` | GREEN |
| SEP-01 | `Dolphin.from_pretrained("JusperLee/Dolphin")` called correctly | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_correct_import_path -x` | GREEN |
| SEP-02 | Audio tensor shape is `[1, samples]` (2D, batch=1) | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_run_dolphin_audio_shape -x` | GREEN |
| SEP-03 | Video tensor shape is `[1, 1, T, 88, 88, 1]` (6D) | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_run_dolphin_video_shape -x` | GREEN |
| SEP-04 | `get_lip_frames(tid, start, end)` returns frames in time window | unit | `./venv/bin/python -m pytest tests/unit/test_lip_extractor.py -x` | Partial — check LipExtractor tests |
| SEP-04 | Temporal sync: empty lip_frames when no face visible during speech | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_separate_without_lip_frames_uses_passthrough -x` | ❌ Wave 0 |
| SEP-05 | DOA angle stored and applied in primary user selection | unit | `./venv/bin/python -m pytest tests/unit/test_engagement.py::test_doa_angle_disambiguates_multiple_faces -x` | ❌ Wave 0 |
| SEP-06 | Passthrough returns `separation_confidence=0.0` when model unavailable | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_passthrough_returns_mono -x` | GREEN |
| SEP-06 | Passthrough triggered when Dolphin raises exception | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_dolphin_exception_falls_back_to_passthrough -x` | ❌ Wave 0 |
| AUD-05 | VAD emits SpeechSegment after silence threshold | unit | `./venv/bin/python -m pytest tests/unit/test_audio_pipeline.py::test_vad_emits_segment_after_silence -x` | ❌ Wave 0 |
| AUD-05 | Ring buffer extracts raw audio aligned to speech window | unit | `./venv/bin/python -m pytest tests/unit/test_audio_pipeline.py::test_ring_buffer_extract_aligned -x` | ❌ Wave 0 |
| AUD-05 | Short segments (<0.5s) are rejected | unit | `./venv/bin/python -m pytest tests/unit/test_audio_pipeline.py::test_short_segment_rejected -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py tests/unit/test_engagement.py -x -q`
- **Per wave merge:** `./venv/bin/python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

The following test files/cases must be created before implementation tasks:

- [ ] `tests/unit/test_audio_pipeline.py` — covers AUD-05 (VAD segmentation, ring buffer, silence threshold, short segment rejection)
- [ ] `tests/unit/test_dolphin_separator.py::test_separate_without_lip_frames_uses_passthrough` — covers SEP-04/SEP-06 interaction
- [ ] `tests/unit/test_dolphin_separator.py::test_dolphin_exception_falls_back_to_passthrough` — covers SEP-06 error path
- [ ] `tests/unit/test_engagement.py::test_doa_angle_disambiguates_multiple_faces` — covers SEP-05

---

## Sources

### Primary (HIGH confidence — directly read from source)

- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/look2hear/models/dolphin.py` — `Dolphin.forward(input, mouth)` signature, `__init__` params, output shape `audio.unsqueeze(dim=1)` = `[batch, 1, samples]`
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/look2hear/models/__init__.py` — `from .dolphin import Dolphin` confirmed
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/perception/dolphin_separator.py` — complete current implementation verified
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/sensors/audio_pipeline.py` — VAD logic, RawAudioBuffer, ring buffer alignment, silence thresholds
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/perception/engagement.py` — DOA update handler, `_select_primary_user` scoring
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/main.py` — audio routing logic in `on_speech_segment` (bug identified)
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/tests/unit/test_dolphin_separator.py` — 6 existing tests, all GREEN
- `.planning/research/ARCHITECTURE.md` — verified Dolphin API patterns from HF Space Inference.py
- `.planning/research/PITFALLS.md` — verified pitfall list with sources
- `.planning/research/STACK.md` — verified stack with confidence levels

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — prior phase decisions, audio routing decisions documented
- `.planning/research/ARCHITECTURE.md` — Dolphin 4-second window, overlap-add pattern

---

## Metadata

**Confidence breakdown:**
- SEP-01 (import API): HIGH — directly verified from `look2hear/models/__init__.py` and `dolphin.py`
- SEP-02 (audio shape): HIGH — `Dolphin.forward(self, input, mouth)` signature read directly
- SEP-03 (video shape): HIGH — existing tests GREEN, shape verified
- SEP-04 (temporal sync): HIGH — both timestamp sources verified as `time.monotonic()`
- SEP-05 (DOA integration): MEDIUM — DOA event handling exists; angle-to-face mapping is new code needing face bbox access (open question)
- SEP-06 (fallback): HIGH — passthrough path tested and GREEN
- AUD-05 (VAD segments): HIGH — full AudioPipeline implementation read; bug in `_reset_speech` identified

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable code, 30-day validity)
