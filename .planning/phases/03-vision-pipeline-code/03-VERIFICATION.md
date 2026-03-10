---
phase: 03-vision-pipeline-code
verified: 2026-03-10T08:30:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 3: Vision Pipeline Code Verification Report

**Phase Goal:** Fix lip_roi_size config for Dolphin compatibility and write comprehensive unit tests for all four vision modules
**Verified:** 2026-03-10T08:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

The phase goal was: "Lock every VIS-* requirement with TDD tests before touching implementation."

All four VIS-* requirements are now covered by passing tests that prove the production behavior contracts are met.

### Observable Truths

| #  | Truth                                                                          | Status     | Evidence                                                                                    |
|----|--------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------|
| 1  | LipExtractor produces 88x88 RGB mouth ROI from MediaPipe landmarks by default  | VERIFIED   | `test_lip_roi_output_shape` passes; `roi.image.shape == (88, 88, 3)` asserted and passing   |
| 2  | LipExtractor buffers lip frames and retrieves them by time window              | VERIFIED   | `test_get_lip_frames_time_filter`, `test_get_recent_frames`, `test_buffer_appends_on_extract` all pass |
| 3  | LipExtractor cleans up buffer on FACE_LOST event                               | VERIFIED   | `test_face_lost_clears_buffer` passes; buffer confirmed empty after `EventType.FACE_LOST` emitted |
| 4  | GazeEstimator parses L2CS step() results correctly (yaw/pitch attributes)      | VERIFIED   | `test_l2cs_step_result_parsed` passes; yaw=15.0 pitch=-8.0 parsed, is_looking_at_robot=True |
| 5  | GazeEstimator falls back to head pose on empty L2CS result                     | VERIFIED   | `test_l2cs_step_empty_result_falls_back` passes; empty yaw/pitch triggers head pose fallback |
| 6  | GazeEstimator falls back to head pose on L2CS exception                        | VERIFIED   | `test_l2cs_step_exception_falls_back` passes; RuntimeError triggers head pose fallback       |
| 7  | EngagementDetector transitions IDLE -> APPROACHING -> ENGAGED after >2s gaze   | VERIFIED   | `test_sustained_gaze_reaches_engaged` passes; state==ENGAGED after 2.1s at 0.1s intervals  |
| 8  | EngagementDetector resets to IDLE when gaze breaks during APPROACHING          | VERIFIED   | `test_gaze_break_resets_to_idle` passes; single gaze=False call returns state to IDLE       |
| 9  | EngagementDetector walking-past filter suppresses rapidly-moving faces          | VERIFIED   | `test_walking_past_not_engaged` passes; `_is_walking_past()` asserted True, state stays IDLE |
| 10 | FaceTracker IOU computation correct for identical, partial, non-overlapping bboxes | VERIFIED | `test_iou_identical_bboxes` (1.0), `test_iou_no_overlap` (0.0), `test_iou_partial_overlap` (0<x<1), `test_iou_symmetric` all pass |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact                                | Expected                                      | Status     | Details                                    |
|-----------------------------------------|-----------------------------------------------|------------|--------------------------------------------|
| `smait/core/config.py`                  | VisionConfig with `lip_roi_size=(88, 88)`     | VERIFIED   | Line 61: `lip_roi_size: tuple = (88, 88)`  |
| `tests/unit/test_lip_extractor.py`      | LipExtractor unit tests, min 60 lines         | VERIFIED   | 198 lines, 9 tests, all pass               |
| `tests/unit/test_gaze.py`              | GazeEstimator tests including L2CS mock tests | VERIFIED   | 168 lines, 8 tests (5 pre-existing + 3 new), all pass |
| `tests/unit/test_engagement.py`         | EngagementDetector state machine tests, min 80 lines | VERIFIED | 286 lines, 10 tests, all pass       |
| `tests/unit/test_face_tracker.py`       | FaceTracker IOU and event tests, min 50 lines | VERIFIED   | 230 lines, 10 tests, all pass              |

### Key Link Verification

| From                                 | To                                      | Via                                             | Status  | Details                                                                     |
|--------------------------------------|-----------------------------------------|-------------------------------------------------|---------|-----------------------------------------------------------------------------|
| `smait/perception/lip_extractor.py`  | `smait/core/config.py`                  | `config.vision.lip_roi_size`                    | WIRED   | Line 49: `self._roi_size = config.vision.lip_roi_size`                      |
| `tests/unit/test_lip_extractor.py`   | `smait/perception/lip_extractor.py`     | `from smait.perception.lip_extractor import`    | WIRED   | Line 14: `from smait.perception.lip_extractor import LipExtractor, LipROI`  |
| `tests/unit/test_engagement.py`      | `smait/perception/engagement.py`        | `from smait.perception.engagement import`       | WIRED   | Line 18-21: imports `EngagementDetector`, `EngagementState`                 |
| `tests/unit/test_face_tracker.py`    | `smait/perception/face_tracker.py`      | `from smait.perception.face_tracker import`     | WIRED   | Line 20: `from smait.perception.face_tracker import FaceTrack, FaceTracker` |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                       | Status    | Evidence                                                                              |
|-------------|-------------|-----------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------|
| VIS-01      | 03-01-PLAN  | L2CS-Net gaze estimation with correct arch (ResNet50)                            | SATISFIED | `test_correct_arch_param` verifies `arch='ResNet50'` via mock; `test_l2cs_step_result_parsed` verifies step() result parsing |
| VIS-02      | 03-01-PLAN  | Lip extraction produces 88x88 ROI compatible with Dolphin                        | SATISFIED | `test_lip_roi_output_shape` asserts `roi.image.shape == (88, 88, 3)`; `VisionConfig.lip_roi_size = (88, 88)` confirmed in source |
| VIS-03      | 03-02-PLAN  | Gaze-based engagement detection with sustained gaze threshold (>2s)              | SATISFIED | `test_sustained_gaze_reaches_engaged`, `test_gaze_break_resets_to_idle`, `test_disengage_after_gaze_timeout` all pass |
| VIS-04      | 03-02-PLAN  | Face tracking maintains persistent IDs across frames (existing MediaPipe + IOU)  | SATISFIED | `test_match_and_update_reuses_track_id` confirms IOU >= 0.3 reuses track_id; `test_iou_*` tests verify IOU math |

**Requirements status in REQUIREMENTS.md:** VIS-01 through VIS-04 are all marked `[x]` (complete).

**Orphaned requirements check:** No Phase 3 requirements in REQUIREMENTS.md traceability table that are absent from plans. VIS-01/02 claimed by 03-01, VIS-03/04 claimed by 03-02. No orphans.

### Anti-Patterns Found

| File                                           | Line    | Pattern    | Severity | Impact                                                    |
|------------------------------------------------|---------|------------|----------|-----------------------------------------------------------|
| `smait/perception/lip_extractor.py`            | 116, 126 | `return []` | INFO    | Legitimate early-return guards for unknown track_id — not stubs. Both verified by `test_get_lip_frames_empty_for_unknown_track`. |

No blockers or warnings found. No TODO/FIXME/placeholder comments in any modified file.

### Human Verification Required

None. All behaviors are fully verifiable via automated tests:
- ROI shape/dtype: asserted directly on numpy array
- Buffer operations: tested with concrete timestamps and counts
- State machine transitions: driven by deterministic update() calls
- Event emission: captured via subscribe() callbacks
- IOU math: tested against known geometric values

### Commits Verified

All four task commits documented in SUMMARY files confirmed present in git history:

| Commit    | Message                                                      | Files Changed           |
|-----------|--------------------------------------------------------------|-------------------------|
| `a9d83d3` | feat(03-01): fix lip_roi_size default to 88x88 and add LipExtractor tests | `smait/core/config.py`, `tests/unit/test_lip_extractor.py` |
| `acfe3dd` | test(03-01): add L2CS step() mock tests to GazeEstimator    | `tests/unit/test_gaze.py`                                   |
| `c8ea728` | test(03-02): EngagementDetector state machine unit tests     | `tests/unit/test_engagement.py`                             |
| `ee6d9e5` | test(03-02): FaceTracker IOU matching and event unit tests   | `tests/unit/test_face_tracker.py`                           |

### Test Suite Summary

```
37 tests collected
37 passed in 2.51s
0 failed, 0 errors, 0 skipped
```

Breakdown:
- `test_lip_extractor.py`: 9 tests (ROI shape, dtype, None/short landmarks, time filter, unknown track, recent frames, FACE_LOST, buffer append)
- `test_gaze.py`: 8 tests (head pose fallback, looking/not-looking, arch param, 3 L2CS step() mock tests, install instruction)
- `test_engagement.py`: 10 tests (initial state, IDLE->ENGAGED, gaze break, area threshold, ENGAGEMENT_START event, disengage timeout, ENGAGEMENT_LOST event, walking-past filter, DOA bonus, reset)
- `test_face_tracker.py`: 10 tests (IOU identical/no-overlap/partial/symmetric, head pose valid/short, new track creation, track ID reuse, FACE_DETECTED event, FACE_LOST timeout)

### Gaps Summary

No gaps. All must-haves verified, all artifacts substantive and wired, all requirements satisfied, all tests pass.

---

_Verified: 2026-03-10T08:30:00Z_
_Verifier: Claude (gsd-verifier)_
