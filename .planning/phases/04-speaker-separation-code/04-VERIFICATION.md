---
phase: 04-speaker-separation-code
verified: 2026-03-10T09:00:00Z
status: passed
score: 8/8 must-haves verified
re_verification: false
---

# Phase 04: Speaker Separation Code Verification Report

**Phase Goal:** Dolphin separator rewritten with correct look2hear API, mono audio input, grayscale lip video input
**Verified:** 2026-03-10T09:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                    | Status     | Evidence                                                                       |
|----|--------------------------------------------------------------------------|------------|--------------------------------------------------------------------------------|
| 1  | Dolphin is never called without lip frames — passthrough is used instead | VERIFIED   | `if not lip_frames:` at line 115 in dolphin_separator.py; test GREEN          |
| 2  | CAE mono audio is always passed to Dolphin, never raw 4-channel          | VERIFIED   | `segment.cae_audio, channels=1` at line 228-232 in main.py                    |
| 3  | Dolphin exception during separation triggers passthrough fallback         | VERIFIED   | `except Exception: ... return self._passthrough(...)` at lines 121-123        |
| 4  | Model mock output shape matches real Dolphin output [1, 1, samples]      | VERIFIED   | `torch.zeros(1, 1, 16000)` in tests; `squeeze()` produces 1D float32          |
| 5  | VAD emits a SpeechSegment after sufficient silence following speech       | VERIFIED   | `test_vad_emits_segment_after_silence` PASSES in test_audio_pipeline.py        |
| 6  | Short speech segments (<0.5s) are rejected and not emitted               | VERIFIED   | `test_short_segment_rejected` PASSES; MIN_SEGMENT_DURATION_S enforced         |
| 7  | Ring buffer extracts raw audio aligned to the speech time window         | VERIFIED   | `test_ring_buffer_write_and_extract` PASSES; `test_ring_buffer_overrun_returns_none` PASSES |
| 8  | DOA angle disambiguates faces by angular proximity, not flat bonus       | VERIFIED   | `_doa_score_for_face()` implemented; flat `1.2x` removed; disambiguation test PASSES |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact                                   | Expected                                                        | Status   | Details                                                    |
|--------------------------------------------|-----------------------------------------------------------------|----------|------------------------------------------------------------|
| `smait/perception/dolphin_separator.py`    | Early exit passthrough, inference_mode                          | VERIFIED | `if not lip_frames` at line 115; `torch.inference_mode()` at line 175; 224 lines |
| `smait/main.py`                            | CAE mono routing — `segment.cae_audio`                         | VERIFIED | `segment.cae_audio, channels=1` at lines 228-232          |
| `tests/unit/test_dolphin_separator.py`     | Passthrough, exception fallback, output shape tests; min 180 lines | VERIFIED | 306 lines; 10 tests, all pass                             |
| `tests/unit/test_audio_pipeline.py`        | VAD segmentation, ring buffer, mic gating; min 80 lines        | VERIFIED | 239 lines; 6 tests, all pass                              |
| `tests/unit/test_engagement.py`            | DOA disambiguation test (`test_doa_angle_disambiguates`)       | VERIFIED | 373 lines; test exists and passes                         |
| `smait/perception/engagement.py`           | `_doa_score_for_face` with bbox center X and camera FOV        | VERIFIED | Method at lines 238-271; applied in `_select_primary_user` at line 287 |

### Key Link Verification

| From                                  | To                                         | Via                                                  | Status   | Details                                                          |
|---------------------------------------|--------------------------------------------|------------------------------------------------------|----------|------------------------------------------------------------------|
| `smait/main.py`                       | `smait/perception/dolphin_separator.py`    | `separate(segment.cae_audio, lip_frames, channels=1)` | WIRED    | Line 228-232 in main.py; `segment.cae_audio` confirmed present  |
| `smait/perception/dolphin_separator.py` | `_passthrough`                           | Early exit when `lip_frames` empty                   | WIRED    | `if not lip_frames:` at line 115; returns `_passthrough(...)` immediately |
| `smait/perception/engagement.py`      | `smait/perception/face_tracker.py`         | `FaceTrack.bbox` center X for DOA alignment          | WIRED    | `x, _y, w, _h = track.bbox` at line 262; `_doa_score_for_face` called in `_select_primary_user` |
| `tests/unit/test_audio_pipeline.py`   | `smait/sensors/audio_pipeline.py`          | `AudioPipeline.process_cae_audio` + `RawAudioBuffer.extract` | WIRED | Both imported and exercised; all 6 tests pass                   |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                 | Status    | Evidence                                                                                     |
|-------------|-------------|-----------------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------|
| SEP-01      | 04-01       | Dolphin AV-TSE loaded with correct API (`from look2hear.models import Dolphin`) | SATISFIED | `from look2hear.models import Dolphin` at line 74; `test_look2hear_importable` PASSES        |
| SEP-02      | 04-01       | Audio input preprocessed to mono `[1, samples]` at 16kHz for Dolphin       | SATISFIED | `audio_tensor = torch.from_numpy(mono_float32).unsqueeze(0)` at line 152; shape test PASSES |
| SEP-03      | 04-01       | Lip frames preprocessed to 88x88 grayscale at 25fps for Dolphin            | SATISFIED | `cv2.COLOR_RGB2GRAY`, `cv2.resize(gray, (88, 88))` at lines 160-162; video shape test PASSES; tensor shape `[1,1,T,88,88,1]` verified |
| SEP-04      | 04-01       | Audio-visual temporal sync via server-side monotonic timestamps             | SATISFIED | `lip_extractor.get_lip_frames(track_id, segment.start_time, segment.end_time)` at lines 214-218 in main.py |
| SEP-05      | 04-02       | DOA angles integrated into engagement detector for multi-speaker disambiguation | SATISFIED | `_doa_score_for_face()` replaces flat 1.2x bonus; `test_doa_angle_disambiguates_multiple_faces` PASSES |
| SEP-06      | 04-01       | Fallback to CAE passthrough audio when Dolphin is unavailable               | SATISFIED | `if not self._available or self._model is None: return self._passthrough(...)` at lines 111-113 |
| AUD-05      | 04-02       | Silero VAD segments speech from CAE audio with ring buffer alignment        | SATISFIED | `test_vad_emits_segment_after_silence`, `test_short_segment_rejected`, `test_ring_buffer_write_and_extract` all PASS |

All 7 requirements satisfied. No orphaned requirements found.

### Anti-Patterns Found

None. Scanned `smait/perception/dolphin_separator.py`, `smait/perception/engagement.py`, `smait/main.py`, and test files for TODO/FIXME/placeholder/empty returns. No issues detected.

### Human Verification Required

None required. All critical behaviors are verifiable programmatically.

Informational note for future integration testing: The lip frame 25fps alignment (`get_lip_frames` with monotonic timestamps) cannot be fully end-to-end tested without a live video stream. The unit tests verify the contract at the interface level; real-world temporal sync under load is observable only when the full pipeline is running.

### Commits Verified

| Hash    | Message                                                                         |
|---------|---------------------------------------------------------------------------------|
| cec2044 | test(04-01): add failing tests for passthrough-on-empty-lip-frames and inference_mode |
| aeb4453 | feat(04-01): fix DolphinSeparator and main.py audio routing                    |
| 66c3f58 | test(04-02): add AudioPipeline tests and DOA disambiguation test (RED)         |
| a2cb2d2 | feat(04-02): implement per-face DOA angular proximity scoring in EngagementDetector |

All 4 commits confirmed in git history on branch `v3`.

### Test Results

```
tests/unit/test_dolphin_separator.py  — 10/10 PASSED
tests/unit/test_audio_pipeline.py     —  6/6  PASSED
tests/unit/test_engagement.py         — 11/11 PASSED
Full suite: 91 passed, 2 warnings
```

The 2 warnings are third-party `beartype` PEP 585 deprecation notices in the vendored `look2hear` library; not caused by phase 04 changes and not blocking.

### Gaps Summary

No gaps. All must-haves from both plans (04-01 and 04-02) are verified against the actual codebase. The phase goal — Dolphin separator rewritten with correct look2hear API, mono audio input, grayscale lip video input — is fully achieved.

---

_Verified: 2026-03-10T09:00:00Z_
_Verifier: Claude (gsd-verifier)_
