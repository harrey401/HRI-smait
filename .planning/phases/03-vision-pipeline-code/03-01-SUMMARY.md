---
phase: 03-vision-pipeline-code
plan: 01
subsystem: testing
tags: [mediapipe, lip-extractor, gaze, l2cs, lip-roi, vision, tdd]

# Dependency graph
requires:
  - phase: 01-dependency-setup-stub-api-fixes
    provides: LipExtractor, GazeEstimator stubs implemented with correct logic
provides:
  - VisionConfig.lip_roi_size fixed to (88,88) for Dolphin AV-TSE compatibility
  - tests/unit/test_lip_extractor.py with 9 LipExtractor tests (ROI shape/dtype, buffer ops, FACE_LOST)
  - tests/unit/test_gaze.py expanded with 3 L2CS step() mock tests
affects: [04-dolphin-avsep-code, lab-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD RED/GREEN: write failing test first, fix config, verify shape assertion"
    - "Synthetic frame helper: np.zeros((480,640,3)) + white mouth region for ROI extraction"
    - "_make_face_track() helper: normalized lip landmarks at (0.5,0.6) for deterministic bounding box"
    - "MagicMock._l2cs_pipeline injection: replace pipeline attr directly for unit-testing gaze paths"

key-files:
  created:
    - tests/unit/test_lip_extractor.py
  modified:
    - smait/core/config.py
    - tests/unit/test_gaze.py

key-decisions:
  - "VisionConfig.lip_roi_size default corrected to (88,88) — Dolphin requires exactly 88x88 grayscale input; 96x96 would silently produce wrong-shaped tensors"
  - "GazeEstimator L2CS tests went straight to GREEN — production code already handles empty yaw/pitch lists and RuntimeError via _estimate_l2cs fallback logic"

patterns-established:
  - "FakeTrack helper pattern: minimal dataclass with only the fields used by the SUT avoids importing MediaPipe in unit tests"
  - "Buffer verification pattern: extract N frames, call get_lip_frames/get_recent_frames, assert count and timestamps"

requirements-completed: [VIS-01, VIS-02]

# Metrics
duration: 2min
completed: 2026-03-10
---

# Phase 03 Plan 01: Vision Pipeline TDD — LipExtractor and GazeEstimator L2CS Tests Summary

**VisionConfig.lip_roi_size bug fixed (96->88) with TDD proof, plus 3 L2CS step() mock tests covering parse, empty-result fallback, and exception fallback**

## Performance

- **Duration:** 1m 31s
- **Started:** 2026-03-10T07:53:18Z
- **Completed:** 2026-03-10T07:54:49Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Fixed `VisionConfig.lip_roi_size` default from `(96, 96)` to `(88, 88)` — Dolphin AV-TSE requires exactly 88x88 lip frames
- Created `tests/unit/test_lip_extractor.py` with 9 tests covering ROI shape, dtype, None/insufficient landmarks, time-filtered buffer retrieval, recent-frames slice, FACE_LOST cleanup, and append-on-extract
- Added 3 L2CS step() mock tests to `tests/unit/test_gaze.py` covering successful parse (yaw/pitch), empty result fallback, and RuntimeError fallback — all 8 gaze tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix lip_roi_size default and write LipExtractor tests** - `a9d83d3` (feat)
2. **Task 2: Add L2CS step() mock tests to GazeEstimator** - `acfe3dd` (test)

_Note: Task 1 followed full TDD cycle — RED phase confirmed (96,96,3) shape failure, GREEN phase fixed config._

## Files Created/Modified

- `smait/core/config.py` - Changed `VisionConfig.lip_roi_size` default from `(96, 96)` to `(88, 88)`
- `tests/unit/test_lip_extractor.py` - 9 new tests for LipExtractor ROI extraction and buffer operations
- `tests/unit/test_gaze.py` - 3 new tests for L2CS step() result parsing and fallback paths

## Decisions Made

- VisionConfig.lip_roi_size corrected to (88,88): Dolphin requires exactly 88x88 grayscale input; 96x96 would silently produce wrong-shaped tensors downstream
- GazeEstimator L2CS tests went straight to GREEN: production code already handles empty yaw/pitch lists and RuntimeError through the existing _estimate_l2cs fallback logic — no production changes needed

## Deviations from Plan

None - plan executed exactly as written. Both tasks followed the specified TDD pattern.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VIS-01 (L2CS gaze integration tested) and VIS-02 (Dolphin-compatible 88x88 lip ROI) requirements satisfied
- LipExtractor produces correctly-shaped RGB frames for Dolphin; buffer operations fully tested
- Ready for Phase 04 Dolphin AV-TSE integration which ingests lip frames from this extractor

## Self-Check: PASSED

- tests/unit/test_lip_extractor.py: FOUND
- smait/core/config.py: FOUND
- tests/unit/test_gaze.py: FOUND
- 03-01-SUMMARY.md: FOUND
- commit a9d83d3: FOUND
- commit acfe3dd: FOUND

---
*Phase: 03-vision-pipeline-code*
*Completed: 2026-03-10*
