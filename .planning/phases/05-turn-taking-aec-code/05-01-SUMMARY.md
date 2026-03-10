---
phase: 05-turn-taking-aec-code
plan: 01
subsystem: asr
tags: [vad, eou, nemo, parakeet, hallucination-filter, turn-taking]

# Dependency graph
requires:
  - phase: 04-speaker-separation-code
    provides: SeparationResult audio fed to Transcriber

provides:
  - VAD-prob-based EOUDetector.feed_vad_prob() with 1800ms silence threshold
  - NeMo Hypothesis word_confidence extraction path with 0.65 fallback
  - Expanded HALLUCINATION_PHRASES blocklist (+10 phrases)
  - Comprehensive test coverage for EOU and transcriber filtering

affects:
  - 05-02 (AEC/barge-in plan uses barge_in_min_speech_ms from EOUConfig)
  - LAB phases (GPU validation of feed_vad_prob integration with live VAD model)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - VAD hysteresis: enter >= 0.50, exit < 0.35, zone 0.35-0.50 no state change
    - Sample-accurate silence counting: silence_sample_count += n_samples per chunk
    - NeMo confidence extraction: word_confidence -> score -> 0.65 fallback chain

key-files:
  created:
    - tests/unit/test_transcriber.py
  modified:
    - smait/core/config.py
    - smait/perception/eou_detector.py
    - smait/perception/asr.py
    - smait/perception/transcriber.py
    - tests/unit/test_eou_detector.py

key-decisions:
  - "Hallucination filter runs before short confidence check — recognised phrases must always carry hallucination_phrase label"
  - "_extract_confidence handles both list[Hypothesis] and single Hypothesis objects from NeMo transcribe()"
  - "hard_cutoff_ms default changed from 1500 to 1800 to match vad_silence_ms threshold"

patterns-established:
  - "EOUDetector dual-path: feed_vad_prob() primary, on_silence()/predict() fallback"
  - "_emit_end_of_turn() resets both heuristic state and VAD state atomically"

requirements-completed: [ASR-02, ASR-03]

# Metrics
duration: 4min
completed: 2026-03-10
---

# Phase 05 Plan 01: EOUDetector VAD-Based Rewrite and NeMo Confidence Extraction Summary

**VAD-prob silence tracking (1800ms / 28800-sample threshold) added to EOUDetector.feed_vad_prob(), NeMo hypothesis word_confidence extraction wired into ParakeetASR, and hallucination blocklist expanded from 14 to 24 phrases.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-10T08:40:43Z
- **Completed:** 2026-03-10T08:44:53Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- EOUDetector gains feed_vad_prob() with Option B hysteresis (enter >=0.50 / exit <0.35 / zone no-op)
- Sample-accurate silence counting fires END_OF_TURN after exactly 28800 samples (1800ms at 16kHz)
- ParakeetASR._extract_confidence() reads NeMo Hypothesis.word_confidence, falls back to score, then 0.65
- HALLUCINATION_PHRASES expanded with 10 YouTube-artifact phrases (subscribers, like and subscribe, etc.)
- Filter order corrected: hallucination check precedes short-confidence check
- 28 tests across both files — all green; full suite 108 passed

## Task Commits

1. **Task 1 RED: add failing VAD-based EOU tests** - `04640e5` (test)
2. **Task 1 GREEN: rewrite EOUDetector with VAD-prob-based silence tracking** - `1c6c762` (feat)
3. **Task 2 RED: add failing hallucination filter and NeMo confidence tests** - `43e953a` (test)
4. **Task 2 GREEN: NeMo hypothesis confidence extraction + expanded hallucination blocklist** - `8e5a3ac` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `smait/core/config.py` - Added vad_silence_ms=1800, barge_in_min_speech_ms=200 to EOUConfig; hard_cutoff_ms default 1500 -> 1800
- `smait/perception/eou_detector.py` - Full rewrite: feed_vad_prob() primary path, heuristic fallback preserved, _emit_end_of_turn() resets both state domains, reset() clears VAD fields
- `smait/perception/asr.py` - return_hypotheses=True; new _extract_confidence(); _compute_confidence() kept as deprecated
- `smait/perception/transcriber.py` - HALLUCINATION_PHRASES expanded (+10 entries); filter order corrected (hallucination before short-conf)
- `tests/unit/test_eou_detector.py` - 6 new VAD tests added (17 total); all pass
- `tests/unit/test_transcriber.py` - Created with 11 tests; all pass

## Decisions Made

- **Hallucination filter order:** Reordered `_check_filters()` so hallucination phrase check runs before short-confidence check. A phrase in the blocklist must always be labelled `hallucination_phrase` regardless of word count — the prior order caused "thanks for watching" (3 words) to be labelled `low_confidence_short` instead.
- **_extract_confidence() dual-mode:** The method accepts both `list[Hypothesis]` and a single `Hypothesis` directly, since tests pass single objects and production passes lists. Unwrapping logic branches on `isinstance(hyps, (list, tuple))`.
- **hard_cutoff_ms 1500 -> 1800:** Aligned with vad_silence_ms so the wall-clock fallback path matches the sample-accurate primary path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Hallucination filter ran after short-confidence filter**
- **Found during:** Task 2 (test_hallucination_phrase_rejected)
- **Issue:** "thanks for watching" (3 words, conf=0.30) was labelled `low_confidence_short` instead of `hallucination_phrase` because the short-confidence check ran first
- **Fix:** Swapped filter order in `_check_filters()` — hallucination check first, short-confidence second
- **Files modified:** smait/perception/transcriber.py
- **Verification:** test_hallucination_phrase_rejected passes; test_short_low_conf_rejected still passes for non-phrase short utterances
- **Committed in:** 8e5a3ac (Task 2 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in filter ordering)
**Impact on plan:** Fix required for correct labelling of hallucination events. No scope creep.

## Issues Encountered

None beyond the filter-ordering bug documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- EOUDetector.feed_vad_prob() ready for AudioPipeline wiring in Plan 02
- barge_in_min_speech_ms=200 available in EOUConfig for AEC anti-echo delay
- NeMo confidence extraction ready for GPU validation in Phase 7

---
*Phase: 05-turn-taking-aec-code*
*Completed: 2026-03-10*
