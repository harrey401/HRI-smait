---
phase: 04-speaker-separation-code
plan: "01"
subsystem: dolphin-separator
tags: [dolphin, audio-routing, speaker-separation, tdd, passthrough]
dependency_graph:
  requires: [smait/perception/dolphin_separator.py, smait/main.py, smait/sensors/audio_pipeline.py]
  provides: [correct-dolphin-audio-routing, passthrough-on-empty-lip-frames, inference-mode]
  affects: [smait/main.py, tests/unit/test_dolphin_separator.py]
tech_stack:
  added: []
  patterns: [early-exit-guard, tdd-red-green, inference-mode-over-no-grad]
key_files:
  created: []
  modified:
    - smait/perception/dolphin_separator.py
    - smait/main.py
    - tests/unit/test_dolphin_separator.py
decisions:
  - "[04-01]: Early exit passthrough before _run_dolphin when lip_frames=[] — prevents audio-only Dolphin calls that would raise TypeError"
  - "[04-01]: torch.inference_mode() replaces torch.no_grad() — faster, disables grad tracking entirely"
  - "[04-01]: main.py always passes segment.cae_audio to Dolphin with channels=1 — raw 4-channel is wrong input for Dolphin mono model"
  - "[04-01]: Remove video_tensor=None branch from _run_dolphin — separate() now guarantees non-empty lip_frames before _run_dolphin is reached"
metrics:
  duration: 2m42s
  completed_date: "2026-03-10T08:28:00Z"
  tasks_completed: 2
  files_modified: 3
---

# Phase 04 Plan 01: DolphinSeparator Bug Fixes and Audio Routing Summary

**One-liner:** Early passthrough exit on empty lip_frames + CAE-mono-always routing + torch.inference_mode replacing no_grad in DolphinSeparator.

## What Was Built

Fixed two production crash bugs in the speaker separation pipeline:

1. **DolphinSeparator crash prevention**: `Dolphin.forward()` requires both audio and video tensors — calling it without video raises `TypeError`. Added an early exit to `separate()` that returns passthrough when `lip_frames=[]`, before `_run_dolphin` is ever called.

2. **Audio routing correction in main.py**: The previous code passed raw 4-channel audio to Dolphin when available, which is wrong — Dolphin only accepts mono. Fixed to always pass `segment.cae_audio` with `channels=1`.

3. **Performance improvement**: Replaced `torch.no_grad()` with `torch.inference_mode()` in `_run_dolphin()`. Inference mode is faster as it disables grad tracking entirely, not just gradient computation.

4. **Code simplification**: Removed the dead `video_tensor=None` branch from `_run_dolphin()` since `separate()` now guarantees lip_frames is non-empty before reaching that method.

5. **Test expansion**: Added 4 new tests to cover the previously untested paths. Updated existing audio/video shape tests to use 3D mock output `[1, 1, 16000]` matching real Dolphin `audio.unsqueeze(dim=1)` output.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add failing tests (TDD RED) | cec2044 | tests/unit/test_dolphin_separator.py |
| 2 | Fix DolphinSeparator and main.py audio routing | aeb4453 | smait/perception/dolphin_separator.py, smait/main.py |

## TDD Cycle

**RED (Task 1):**
- `test_separate_without_lip_frames_uses_passthrough`: FAILED — model was called even with empty lip_frames
- `test_inference_mode_used`: FAILED — torch.no_grad() used instead
- `test_dolphin_exception_falls_back_to_passthrough`: Already GREEN (try/except existed)
- `test_run_dolphin_output_shape_matches_real_model`: Already GREEN (squeeze handles 3D)

**GREEN (Task 2):**
- All 10 DolphinSeparator tests pass

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Pre-existing Failures (Out of Scope)

`test_doa_angle_disambiguates_multiple_faces` in `tests/unit/test_engagement.py` was already failing before this plan's changes. The DOA angular scoring in `EngagementDetector` prefers face area over angular proximity — this is a pre-existing bug not caused by this plan's work.

**Deferred to:** Future engagement detector fix plan.

## Verification Results

- `grep -n "segment.raw_audio" smait/main.py` — NOT found in Dolphin separation path (correct)
- `grep -n "inference_mode" smait/perception/dolphin_separator.py` — found at line 175
- `grep -n "if not lip_frames" smait/perception/dolphin_separator.py` — found at line 115
- All 10 DolphinSeparator tests pass
- 80 tests pass across full suite (excluding pre-existing engagement failure)

## Success Criteria Verification

1. DolphinSeparator.separate() returns passthrough when lip_frames is empty — DONE (line 115-117)
2. main.py always passes segment.cae_audio to Dolphin with channels=1 — DONE
3. Dolphin exception during separation falls back to passthrough — DONE (pre-existing try/except)
4. torch.inference_mode() used instead of torch.no_grad() — DONE (line 175)
5. Mock output shapes in tests match real Dolphin output [1, 1, samples] — DONE
6. All tests pass (new + existing) — DONE (10/10 DolphinSeparator tests GREEN)

## Self-Check: PASSED
