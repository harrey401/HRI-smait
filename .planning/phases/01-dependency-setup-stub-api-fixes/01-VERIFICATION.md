---
phase: 01-dependency-setup-stub-api-fixes
verified: 2026-03-09T00:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 01: Dependency Setup & Stub API Fixes — Verification Report

**Phase Goal:** Install missing dependencies, vendor Dolphin source, fix broken stub files, and bootstrap the test harness so that all unit tests pass on a CPU-only machine.
**Verified:** 2026-03-09
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `from look2hear.models import Dolphin` succeeds on CPU without GPU | VERIFIED | `venv/bin/python3 -c "from look2hear.models import Dolphin; print('OK')"` prints `Dolphin import OK`; look2hear/ exists at project root with models/__init__.py and dolphin.py |
| 2 | pytest discovers and runs tests from tests/ directory | VERIFIED | `venv/bin/python3 -m pytest tests/unit/ -q` reports `28 passed, 2 warnings in 4.58s`; all four test files collected with zero collection errors |
| 3 | EventBus does not cache stale asyncio loop between test runs | VERIFIED | `smait/core/events.py` has no `self._loop` field; `emit()` calls `asyncio.get_running_loop()` on line 92 on every invocation with no caching |
| 4 | DolphinSeparator.init_model() imports from look2hear.models, not dolphin | VERIFIED | Line 74 of dolphin_separator.py: `from look2hear.models import Dolphin`; no reference to `DolphinModel` anywhere in smait/ |
| 5 | DolphinSeparator._run_dolphin() prepares audio as [1, samples] mono and video as [1, 1, T, 88, 88, 1] grayscale | VERIFIED | Lines 148 and 167-169 of dolphin_separator.py implement exact shapes; test_run_dolphin_audio_shape and test_run_dolphin_video_shape both PASS |
| 6 | TTSEngine.init_model() instantiates KPipeline(lang_code='a'), not KokoroTTS() | VERIFIED | Line 54-55 of tts.py: `from kokoro import KPipeline` then `self._pipeline = KPipeline(lang_code="a")`; no reference to `KokoroTTS` anywhere in smait/ |
| 7 | TTSEngine.synthesize() iterates the KPipeline generator and concatenates PCM chunks | VERIFIED | Lines 86-91 of tts.py: `for _graphemes, _phonemes, audio in self._pipeline(...)` with PCM concatenation; test_synthesize_uses_generator PASSES |
| 8 | GazeEstimator.init_model() passes arch='ResNet50' to L2CSPipeline, not 'Gaze360' | VERIFIED | Line 59 of gaze.py: `arch='ResNet50'`; no `Gaze360` anywhere in smait/ |
| 9 | EOUDetector.init_model() does not attempt any LiveKit or HuggingFace model import | VERIFIED | eou_detector.py init_model() body (lines 43-52) only logs and sets `self._available = False`; test_no_livekit_import and test_no_transformers_import both PASS |
| 10 | EOUDetector.predict() always uses heuristic path in Phase 1 | VERIFIED | predict() (lines 93-101) calls `_heuristic_eou()` directly; no model-gated branch exists; 6 heuristic tests all PASS |
| 11 | GazeEstimator still falls back to head pose when L2CS-Net is unavailable | VERIFIED | estimate() (lines 80-82) checks `self._l2cs_pipeline is not None` and routes to `_estimate_from_head_pose()`; test_head_pose_fallback PASSES |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `look2hear/` | Vendored Dolphin look2hear package | VERIFIED | Directory exists; models/__init__.py and dolphin.py present |
| `look2hear/models/__init__.py` | Exportable Dolphin class | VERIFIED | Exists; CPU import succeeds in venv |
| `vendor/dolphin-src/` | Full Dolphin repo clone | VERIFIED | Directory exists; gitignored via `vendor/` in .gitignore line 15 |
| `tests/conftest.py` | Shared fixtures (min 20 lines) | VERIFIED | 39 lines; reset_config_singleton (autouse), config, event_bus, silence_audio, speech_audio fixtures all present |
| `tests/unit/test_dolphin_separator.py` | Test scaffold (min 15 lines) | VERIFIED | 150 lines; 6 tests passing |
| `tests/unit/test_tts.py` | Test scaffold (min 15 lines) | VERIFIED | 100 lines; 6 tests passing |
| `tests/unit/test_gaze.py` | Test scaffold (min 15 lines) | VERIFIED | 107 lines; 5 tests passing |
| `tests/unit/test_eou_detector.py` | Test scaffold (min 15 lines) | VERIFIED | 146 lines; 11 tests passing |
| `smait/perception/dolphin_separator.py` | Corrected Dolphin import and tensor shapes | VERIFIED | Contains `from look2hear.models import Dolphin`; [1,samples] audio and [1,1,T,88,88,1] video shapes confirmed |
| `smait/output/tts.py` | Corrected Kokoro KPipeline integration | VERIFIED | Contains `from kokoro import KPipeline`; generator API used in synthesize() |
| `smait/perception/gaze.py` | Corrected L2CS-Net arch parameter | VERIFIED | Contains `arch='ResNet50'` at line 59; edavalosanaya fork URL in warning |
| `smait/perception/eou_detector.py` | LiveKit import removed, heuristic-only EOU | VERIFIED | Contains `self._available = False` in init_model(); no livekit or transformers imports |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `look2hear/` | `smait/perception/dolphin_separator.py` | Python import resolution | WIRED | `from look2hear.models import Dolphin` on line 74 of dolphin_separator.py; import succeeds at runtime |
| `tests/conftest.py` | `smait/core/config.py` | reset_config fixture | WIRED | `from smait.core.config import Config, reset_config` on line 5 of conftest.py; used in autouse fixture |
| `smait/perception/dolphin_separator.py` | `look2hear/models/__init__.py` | Python import | WIRED | Lazy import inside init_model() resolves to vendored package on CPU |
| `smait/output/tts.py` | `kokoro` | pip package import | WIRED | `from kokoro import KPipeline` line 54; `venv/bin/python3 -c "from kokoro import KPipeline"` succeeds |
| `smait/perception/gaze.py` | `l2cs` | pip package import | WIRED | `from l2cs import Pipeline as L2CSPipeline` line 56; `venv/bin/python3 -c "from l2cs import Pipeline"` succeeds |
| `smait/perception/eou_detector.py` | `smait/core/events.py` | EventBus emit | WIRED | `self._event_bus.emit(EventType.END_OF_TURN, ...)` line 130; EventBus correctly retrieved per-test without loop caching |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ENV-03 | 01-01-PLAN | Dolphin AV-TSE vendored from source (not pip) | SATISFIED | look2hear/ committed to repo; `from look2hear.models import Dolphin` succeeds CPU-only; vendor/dolphin-src/ gitignored |
| QUAL-01 | 01-02-PLAN, 01-03-PLAN | All stub APIs corrected to match real model interfaces | SATISFIED | DolphinModel removed (look2hear.models.Dolphin used); KokoroTTS removed (kokoro.KPipeline used); Gaze360 removed (ResNet50 used); all livekit/transformers imports removed from eou_detector; 28/28 tests pass confirming correct interfaces |

No orphaned requirements detected. Both IDs claimed in plan frontmatter are accounted for and satisfied.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `look2hear/models/video_compoent.py` | 381 | `typing.Tuple[int, int, int]` deprecated (beartype warning) | Info | Deprecation warning from vendored third-party code; not a project concern; does not affect tests or runtime |

No blockers. No stubs. No TODO/FIXME/PLACEHOLDER markers in any modified source files.

---

### Human Verification Required

None. All behaviors verifiable programmatically via the test suite.

---

### Gaps Summary

No gaps. Phase goal fully achieved.

**What was delivered:**
- `look2hear/` vendored at project root; CPU import verified live
- `vendor/dolphin-src/` present and gitignored
- `tests/conftest.py` with 5 fixtures (39 lines, exceeds 20-line minimum)
- 4 unit test scaffolds (150, 100, 107, 146 lines respectively — all exceed 15-line minimum)
- EventBus `self._loop` field removed; `asyncio.get_running_loop()` called per-emit
- `pyproject.toml` dev deps include `pytest-cov>=4.0.0`
- `.gitignore` includes `vendor/`
- All 4 stub files corrected: dolphin_separator.py, tts.py, gaze.py, eou_detector.py
- All fabricated class names eliminated: DolphinModel, KokoroTTS, Gaze360, livekit.plugins.turn_detector, transformers.AutoModelForSequenceClassification
- 6 commits verifiable in git log (11a9f1a, 1d3e9a1, e05756e, f7b5f70, 85200d7, 5b065bf)
- 28 unit tests pass, 0 failures, 0 xfail markers in active code

---

_Verified: 2026-03-09_
_Verifier: Claude (gsd-verifier)_
