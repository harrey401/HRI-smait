---
phase: 01-dependency-setup-stub-api-fixes
plan: 02
subsystem: perception, tts
tags: [dolphin, look2hear, kokoro, tts, audio-separation, tensor-shapes, imports]

# Dependency graph
requires:
  - phase: 01-dependency-setup-stub-api-fixes
    plan: 01
    provides: "look2hear vendored, xfail test scaffolding for dolphin and tts"

provides:
  - "DolphinSeparator using from look2hear.models import Dolphin with correct [1, samples] audio and [1, 1, T, 88, 88, 1] grayscale video tensors"
  - "TTSEngine using from kokoro import KPipeline with generator API and voice config"
  - "All fabricated imports (DolphinModel, KokoroTTS) removed"
  - "12 unit tests passing (6 dolphin, 6 tts) — no xfail markers remaining"

affects:
  - "Phase 4: speaker separation — DolphinSeparator is now ready for real model weight loading"
  - "Phase 2: TTS pipeline — TTSEngine KPipeline integration complete"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mono-first audio: always mix down to [1, samples] before passing to Dolphin"
    - "Grayscale lip frames: RGB->grayscale->88x88->unsqueeze twice to get [1, 1, T, 88, 88, 1]"
    - "KPipeline generator: iterate (graphemes, phonemes, audio) chunks and concatenate PCM bytes"

key-files:
  created: []
  modified:
    - smait/perception/dolphin_separator.py
    - smait/output/tts.py
    - tests/unit/test_dolphin_separator.py
    - tests/unit/test_tts.py

key-decisions:
  - "Dolphin takes mono audio [1, samples] — always average multi-channel input before model call, track multichannel flag separately"
  - "Grayscale at model boundary: convert RGB->grayscale->resize 88x88 inside _run_dolphin, not in caller"
  - "KPipeline voice defaults to 'af_heart' via getattr on TTSConfig — no schema change needed"

patterns-established:
  - "Tensor shape validation: test both audio shape (ndim, batch) and video shape (ndim, all 6 dims) by capturing mock call args"
  - "Patch at library boundary: patch 'look2hear.models.Dolphin' and 'kokoro.KPipeline' — not 'smait.*.Dolphin'"

requirements-completed: [QUAL-01]

# Metrics
duration: 10min
completed: 2026-03-10
---

# Phase 01 Plan 02: Fix Dolphin and Kokoro Stub Imports Summary

**DolphinSeparator rewritten to use `look2hear.models.Dolphin` with correct mono [1, samples] audio and [1, 1, T, 88, 88, 1] grayscale video tensors; TTSEngine rewritten to use `kokoro.KPipeline` generator API — 12 tests passing, zero xfail markers.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-10T01:38:00Z
- **Completed:** 2026-03-10T01:42:37Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Removed fabricated `DolphinModel` import; DolphinSeparator now correctly loads via `Dolphin.from_pretrained("JusperLee/Dolphin")`
- Fixed multi-channel audio handling: always mix down to mono before model call, produce `[1, samples]` tensor (was incorrectly 3D for multichannel)
- Fixed lip video tensor: convert RGB to grayscale, resize to 88x88, produce `[1, 1, T, 88, 88, 1]` (was wrong shape `[1, T, H, W, C]`)
- Removed fabricated `KokoroTTS` import; TTSEngine now correctly uses `KPipeline(lang_code='a')` generator API
- Fixed `synthesize()` to iterate `(graphemes, phonemes, audio)` generator chunks and concatenate PCM bytes
- All xfail markers removed: 12 unit tests pass (6 dolphin, 6 tts)

## Task Commits

1. **Task 1: Fix DolphinSeparator stub imports and tensor shapes** - `e05756e` (feat)
2. **Task 2: Fix TTSEngine stub imports and API usage** - `85200d7` (feat)

## Files Created/Modified

- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/perception/dolphin_separator.py` - Correct look2hear import, mono audio [1,samples], grayscale video [1,1,T,88,88,1]
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/smait/output/tts.py` - Correct KPipeline import, generator iteration, voice config
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/tests/unit/test_dolphin_separator.py` - xfail removed; added test_run_dolphin_audio_shape, test_run_dolphin_video_shape
- `/home/gow/.openclaw/workspace/projects/SMAIT-v3/tests/unit/test_tts.py` - xfail removed; added test_voice_from_config, test_sample_rate_24khz

## Decisions Made

- Dolphin mono-first: always average multi-channel input inside `_run_dolphin` before producing `[1, samples]`. The `used_multichannel` flag is preserved for logging but does not change the model's input shape.
- Grayscale conversion happens at model boundary in `_run_dolphin` (not in LipExtractor), keeping the rest of the pipeline RGB-native.
- `_voice` defaults to `'af_heart'` using `getattr(config.tts, 'voice', 'af_heart')` — no schema change to `TTSConfig` needed at this stage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed LipROI constructor in test — `bbox` and `confidence` fields do not exist**
- **Found during:** Task 1 (writing test_run_dolphin_video_shape)
- **Issue:** Test used `LipROI(image=..., bbox=..., confidence=...)` but `LipROI` only has `image`, `timestamp`, `track_id`
- **Fix:** Updated test to use the correct constructor signature
- **Files modified:** tests/unit/test_dolphin_separator.py
- **Verification:** Test passes after fix
- **Committed in:** e05756e (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test constructor)
**Impact on plan:** Minor fix to test code only. No scope creep.

## Issues Encountered

None — implementation was straightforward once the correct API signatures from RESEARCH.md were applied.

## Next Phase Readiness

- DolphinSeparator: stub is production-ready in terms of import and tensor shapes. Blocked only on actual model weights (Phase 7 lab work)
- TTSEngine: KPipeline integration complete. Ready for Phase 2 TTS streaming pipeline work
- No blockers for Phase 03 (remaining stub fixes in this phase)

## Self-Check: PASSED

- FOUND: smait/perception/dolphin_separator.py
- FOUND: smait/output/tts.py
- FOUND: .planning/phases/01-dependency-setup-stub-api-fixes/01-02-SUMMARY.md
- FOUND: commit e05756e (Task 1: DolphinSeparator fix)
- FOUND: commit 85200d7 (Task 2: TTSEngine fix)
- All 12 tests pass: 6 dolphin, 6 tts

---
*Phase: 01-dependency-setup-stub-api-fixes*
*Completed: 2026-03-10*
