---
phase: 05-turn-taking-aec-code
verified: 2026-03-10T11:00:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 05: Turn-Taking / AEC Verification Report

**Phase Goal:** VAD-based EOU, AEC research, barge-in logic
**Verified:** 2026-03-10T11:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

Plan 01 (ASR-02, ASR-03) truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | VAD silence >= 1800ms after speech triggers END_OF_TURN | VERIFIED | `feed_vad_prob()` accumulates `_silence_sample_count`; fires when >= 28800 samples; `test_vad_silence_triggers_eou` passes |
| 2 | VAD silence < 1800ms does NOT trigger END_OF_TURN | VERIFIED | Guard `>= self._silence_threshold_samples`; `test_vad_short_silence_no_eou` passes |
| 3 | Speech detection resets the silence sample counter | VERIFIED | `speech_prob >= 0.50` branch sets `_silence_sample_count = 0`; `test_vad_speech_resets_counter` passes |
| 4 | Known hallucination phrases at low confidence are rejected | VERIFIED | `_check_filters()` checks HALLUCINATION_PHRASES first; `test_hallucination_phrase_rejected` passes |
| 5 | Phrases above confidence threshold are accepted | VERIFIED | Hallucination check only rejects when `conf < 0.60`; `test_hallucination_phrase_accepted_high_conf` passes |
| 6 | Short utterances with low confidence are rejected | VERIFIED | `word_count < 8 and conf < 0.40` path; `test_short_low_conf_rejected` passes |
| 7 | NeMo Hypothesis.word_confidence extraction path exists with fallback | VERIFIED | `_extract_confidence()` reads `word_confidence` -> `score` -> 0.65; `test_extract_confidence_word_confidence` and `test_extract_confidence_fallback` pass |

Plan 02 (AUD-06, AUD-07) truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 8 | SoftwareAEC.process_near() returns same-length PCM after echo cancellation | VERIFIED | `test_process_near_returns_bytes` passes (512 in -> 512 out) |
| 9 | AEC is skipped when cae_status.aec is True | VERIFIED | Guard `not self._cae_status.get("aec", False)` in `process_cae_audio()`; `test_cae_aec_gating_hardware_aec_active` passes |
| 10 | SoftwareAEC is invoked from AudioPipeline.process_cae_audio() when cae_status.aec is False | VERIFIED | `self._aec.process_near(data)` called in guard block; `test_pipeline_aec_processes_audio` passes |
| 11 | BARGE_IN event fires when VAD detects speech during TTS playback | VERIFIED | `_tts_playing` path emits `EventType.BARGE_IN`; `test_barge_in_emitted_during_tts` passes |
| 12 | TTS task is cancelled on BARGE_IN (not just flagged) | VERIFIED | `_on_barge_in()` calls `self._tts_task.cancel()`; `test_tts_task_cancelled_on_barge_in` passes |
| 13 | TTS_END is always emitted after barge-in cancellation (finally block) | VERIFIED | `finally` block in `speak()` and `speak_streaming()` always calls `emit_async(TTS_END)`; `test_tts_end_after_barge_in` passes |

**Score:** 13/13 truths verified

---

### Required Artifacts

**Plan 01 artifacts:**

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `smait/perception/eou_detector.py` | VAD-prob-based EOU with 1800ms threshold | VERIFIED | Contains `feed_vad_prob`; 240 lines; wired via `EventType.END_OF_TURN` emission |
| `smait/perception/transcriber.py` | Hallucination filtering with expanded blocklist | VERIFIED | Contains `HALLUCINATION_PHRASES` (24 phrases); wired to `ParakeetASR` |
| `smait/perception/asr.py` | NeMo hypothesis confidence extraction | VERIFIED | Contains `return_hypotheses=True` in `transcribe()`; `_extract_confidence()` implemented |
| `tests/unit/test_eou_detector.py` | VAD-based EOU test cases | VERIFIED | 263 lines (min: 80); 17 tests; 6 new VAD tests all pass |
| `tests/unit/test_transcriber.py` | Hallucination filter test coverage | VERIFIED | 223 lines (min: 60); 11 tests; all pass |

**Plan 02 artifacts:**

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `smait/sensors/audio_pipeline.py` | Barge-in VAD path, AEC integration | VERIFIED | Contains `_tts_playing`; `process_cae_audio()` emits `BARGE_IN`; calls `_aec.process_near()` |
| `smait/output/tts.py` | Cancellable TTS task with BARGE_IN handler | VERIFIED | Contains `_on_barge_in`; `_tts_task = asyncio.current_task()` in both `speak()` and `speak_streaming()` |
| `smait/core/events.py` | BARGE_IN event type | VERIFIED | `BARGE_IN = auto()` present in Turn-taking section after `END_OF_TURN` |
| `smait/sensors/aec.py` | SoftwareAEC class with speexdsp | VERIFIED | Contains `SoftwareAEC`; lazy speexdsp import; `FRAME_SAMPLES=256`, `FRAME_BYTES=512` |
| `tests/unit/test_aec.py` | SoftwareAEC unit tests | VERIFIED | 242 lines (min: 40); 8 tests; all pass |
| `tests/unit/test_barge_in.py` | Barge-in state transition tests | VERIFIED | 275 lines (min: 60); 7 tests; all pass |

---

### Key Link Verification

**Plan 01 key links:**

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `smait/perception/eou_detector.py` | `smait/core/events.py` | `EventType.END_OF_TURN` emission | WIRED | `_emit_end_of_turn()` calls `self._event_bus.emit(EventType.END_OF_TURN, {...})`; pattern `emit.*END_OF_TURN` present at line 209 |
| `smait/perception/asr.py` | `smait/perception/transcriber.py` | `TranscriptResult` with real confidence | WIRED | `return_hypotheses=True` in `transcribe()` at line 94; `_extract_confidence(result)` called at line 111 |

**Plan 02 key links:**

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `smait/sensors/audio_pipeline.py` | `smait/core/events.py` | `EventType.BARGE_IN` emission on speech during TTS | WIRED | `self._event_bus.emit(EventType.BARGE_IN)` at line 243 inside `_tts_playing` guard |
| `smait/output/tts.py` | `smait/sensors/audio_pipeline.py` | `BARGE_IN` subscription cancels TTS task | WIRED | `_on_barge_in()` at line 82; subscribed via `event_bus.subscribe(EventType.BARGE_IN, self._on_barge_in)` at line 56 |
| `smait/sensors/audio_pipeline.py` | `smait/sensors/aec.py` | `SoftwareAEC.process_near()` called in `process_cae_audio()` gated on `not cae_status.aec` | WIRED | `self._aec.process_near(data)` at line 212; guard `not self._cae_status.get("aec", False)` at line 211 |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ASR-02 | 05-01-PLAN.md | Hallucination filtering rejects phantom transcripts (confidence + phrase blocklist) | SATISFIED | `HALLUCINATION_PHRASES` (24 entries), `_check_filters()` with hallucination-first ordering, 11 transcriber tests all green |
| ASR-03 | 05-01-PLAN.md | VAD-based end-of-utterance with ~1.8s silence threshold | SATISFIED | `feed_vad_prob()` with 28800-sample threshold (1800ms at 16kHz), 6 VAD tests all green |
| AUD-06 | 05-02-PLAN.md | Acoustic echo cancellation replaces mic gating | SATISFIED | `SoftwareAEC` class with speexdsp, wired into `AudioPipeline.process_cae_audio()` gated on `not cae_status.aec`; 8 AEC tests green |
| AUD-07 | 05-02-PLAN.md | Barge-in support — robot listens while speaking, user can interrupt | SATISFIED | `BARGE_IN` event type added, `_tts_playing` path keeps VAD active, 200ms anti-echo guard, `TTSEngine._on_barge_in()` cancels task; 7 barge-in tests green |

No orphaned requirements found. All four requirement IDs (ASR-02, ASR-03, AUD-06, AUD-07) are claimed by plans and verified in code.

---

### Anti-Patterns Found

None. Scan of all 7 modified files found no TODO/FIXME/placeholder comments and no empty stub returns.

---

### Additional Anti-echo Guard Verification

The 200ms anti-echo guard for barge-in was specifically checked:

- `self._barge_in_min_speech_ms` initialized from `config.eou.barge_in_min_speech_ms` (default 200)
- Guard check in `process_cae_audio()`: `elapsed_ms = (time.monotonic() - self._tts_start_time) * 1000; if elapsed_ms < self._barge_in_min_speech_ms: continue`
- `test_barge_in_delay_guard` verifies: at 50ms after TTS_START (within 200ms window), no BARGE_IN emitted

Truth "No barge-in fires within 200ms of TTS_START" — VERIFIED.

---

### Human Verification Required

The following behaviors are correct in code but cannot be verified without hardware:

1. **SoftwareAEC actual echo cancellation quality**
   - Test: Play TTS audio through speaker while recording with mic. Check that mic recording shows reduced echo.
   - Expected: Near-end mic audio shows measurably less TTS echo with speexdsp active vs. passthrough.
   - Why human: Requires `libspeexdsp-dev` + physical speaker/mic setup; cannot run in CI.

2. **Barge-in detection latency on real audio**
   - Test: Robot speaks a sentence; user interrupts mid-sentence. Measure time from user speech onset to TTS stopping.
   - Expected: TTS stops within one VAD chunk (30ms) of user speech exceeding threshold, plus 200ms guard if applicable.
   - Why human: Requires live audio stack with Silero VAD loaded; VAD mocked in unit tests.

3. **False barge-in rate with real TTS audio**
   - Test: Robot speaks for 30 seconds with no human in the room. Count spurious BARGE_IN events.
   - Expected: Zero BARGE_IN events (echo guard prevents TTS echoes from triggering barge-in).
   - Why human: Requires physical environment; dependent on room acoustics and speaker volume.

---

### Commit Verification

All 8 commits documented in SUMMARY files verified to exist in git history:

- `04640e5` (test: failing VAD EOU tests) — EXISTS
- `1c6c762` (feat: VAD-prob EOUDetector rewrite) — EXISTS
- `43e953a` (test: failing hallucination/NeMo confidence tests) — EXISTS
- `8e5a3ac` (feat: NeMo confidence extraction + expanded blocklist) — EXISTS
- `66a19eb` (test: failing barge-in tests) — EXISTS
- `2714ae7` (feat: BARGE_IN event, barge-in VAD path, cancellable TTS) — EXISTS
- `796a9a6` (test: failing SoftwareAEC and pipeline AEC tests) — EXISTS
- `e31fcb5` (feat: SoftwareAEC + AudioPipeline wiring) — EXISTS

TDD pattern confirmed: RED commit precedes GREEN commit for both plans. Full test count: 43 tests across 4 files.

---

## Summary

Phase 05 goal fully achieved. All 13 must-have truths verified through code inspection and passing tests:

- `EOUDetector.feed_vad_prob()` implements sample-accurate 1800ms silence detection with hysteresis (Plan 01 / ASR-03)
- Hallucination filtering with 24-phrase blocklist and correct filter ordering (Plan 01 / ASR-02)
- NeMo `word_confidence` extraction path with `score` fallback and 0.65 default (Plan 01 / ASR-02)
- `SoftwareAEC` with speexdsp, graceful degradation, frame buffering, CAE gating (Plan 02 / AUD-06)
- Barge-in VAD path with 200ms anti-echo guard replacing full mic gate (Plan 02 / AUD-07)
- Cancellable `TTSEngine` via `asyncio.Task.cancel()` with guaranteed `TTS_END` in finally (Plan 02 / AUD-07)

No gaps. No anti-patterns. 43/43 tests pass.

---

_Verified: 2026-03-10T11:00:00Z_
_Verifier: Claude (gsd-verifier)_
