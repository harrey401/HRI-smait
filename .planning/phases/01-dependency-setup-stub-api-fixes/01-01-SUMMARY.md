---
phase: 01-dependency-setup-stub-api-fixes
plan: "01"
subsystem: infrastructure
tags: [vendoring, look2hear, dolphin, kokoro, l2cs, pytest, eventbus, test-infrastructure]
dependency_graph:
  requires: []
  provides:
    - look2hear importable from project root
    - pytest test suite with conftest fixtures
    - EventBus stale loop fixed
    - Kokoro KPipeline pip-installed
    - L2CS-Net Pipeline pip-installed
    - 4 RED test scaffolds (will go GREEN in Plans 02 and 03)
  affects:
    - smait/core/events.py (EventBus bug fix)
    - all subsequent plans (test infrastructure foundation)
tech_stack:
  added:
    - look2hear (vendored from JusperLee/Dolphin, committed)
    - vector-quantize-pytorch==1.27.21
    - taylor-series-linear-attention==0.1.12
    - beartype==0.22.9
    - kokoro>=0.9.4 (KPipeline generator API, 24kHz)
    - l2cs==0.0.1 (edavalosanaya fork, pip-installable)
    - pytest==9.0.2
    - pytest-asyncio==1.3.0
    - pytest-cov==7.0.0
  patterns:
    - Vendoring Python package by copying into project root (portable, git-tracked)
    - xfail markers for RED-phase TDD tests (Plans 02/03 will make green)
    - autouse fixture for config singleton reset between tests
    - asyncio.get_running_loop() instead of caching self._loop
key_files:
  created:
    - look2hear/models/__init__.py
    - look2hear/models/dolphin.py
    - look2hear/models/video_compoent.py
    - look2hear/datas/transform.py
    - tests/__init__.py
    - tests/unit/__init__.py
    - tests/conftest.py
    - tests/unit/test_dolphin_separator.py
    - tests/unit/test_tts.py
    - tests/unit/test_gaze.py
    - tests/unit/test_eou_detector.py
  modified:
    - pyproject.toml (added pytest-cov>=4.0.0 to dev deps)
    - smait/core/events.py (removed stale loop caching)
    - .gitignore (added vendor/ exclusion)
decisions:
  - "Vendor look2hear/ by copy not symlink: git tracks copies; symlinks to external paths break on other machines"
  - "Use xfail markers for stub-correctness tests: they are intended to be RED now and GREEN in Plans 02/03"
  - "Use python3 -m pip instead of venv/bin/pip: pip shebang points to wrong venv python"
metrics:
  duration_minutes: 15
  tasks_completed: 2
  files_created: 11
  files_modified: 3
  completed_date: "2026-03-10"
---

# Phase 01 Plan 01: Dependency Setup and Test Infrastructure Bootstrap Summary

**One-liner:** Vendored look2hear/ from JusperLee/Dolphin into project root, bootstrapped pytest test suite with 4 RED TDD scaffolds, and fixed EventBus stale asyncio loop by replacing self._loop caching with per-call asyncio.get_running_loop().

## What Was Built

### Task 1: Vendor Dolphin and Install Pip Dependencies

- Cloned JusperLee/Dolphin repo into `vendor/dolphin-src/` (gitignored)
- Copied `look2hear/` package into project root and committed it (2MB, portable)
- Verified: `from look2hear.models import Dolphin` succeeds on CPU without downloading weights
- Installed Dolphin's Python dependencies into project venv: `vector-quantize-pytorch`, `taylor-series-linear-attention`, `beartype`, `einops`
- Installed `kokoro>=0.9.4`: `from kokoro import KPipeline` verified
- Installed `l2cs==0.0.1` from edavalosanaya fork: `from l2cs import Pipeline` verified
- Installed `pytest==9.0.2`, `pytest-asyncio==1.3.0`, `pytest-cov==7.0.0`
- Added `pytest-cov>=4.0.0` to `pyproject.toml` dev optional-dependencies
- Added `vendor/` to `.gitignore`

### Task 2: Bootstrap Test Infrastructure and Fix EventBus Stale Loop

**EventBus fix (`smait/core/events.py`):**
- Removed `self._loop: asyncio.AbstractEventLoop | None = None` field
- In `emit()`, replaced `if loop is None: loop = asyncio.get_running_loop(); self._loop = loop` with direct `loop = asyncio.get_running_loop()` on every call
- Result: EventBus works correctly across multiple `asyncio.run()` calls in tests

**Test infrastructure (`tests/`):**
- `tests/conftest.py`: `reset_config_singleton` (autouse), `config`, `event_bus`, `silence_audio`, `speech_audio` fixtures
- `tests/unit/test_dolphin_separator.py`: 3 tests (2 pass + 1 xfail)
- `tests/unit/test_tts.py`: 4 tests (2 pass + 2 xfail)
- `tests/unit/test_gaze.py`: 4 tests (3 pass + 1 xfail)
- `tests/unit/test_eou_detector.py`: 9 tests (8 pass + 1 xfail)

**Final result:** 15 passed, 5 xfailed — no collection errors, no unexpected failures

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pip shebang in SMAIT-v3 venv pointed to wrong Python**

- **Found during:** Task 1 — `venv/bin/pip install` was silently installing into the HRI-smait venv instead
- **Issue:** `venv/bin/pip` had shebang `#!/home/gow/.openclaw/workspace/projects/HRI-smait/venv/bin/python3` — packages installed via `venv/bin/pip` went into the wrong location
- **Fix:** Used `venv/bin/python3 -m pip install` throughout, which correctly uses the SMAIT-v3 venv's Python binary (symlinked to `/usr/bin/python3`)
- **Files modified:** None (fix was in the install commands, not source files)
- **Commit:** 11a9f1a

**2. [Rule 2 - Missing deps] Dolphin dependencies not auto-installed by requirements.txt**

- **Found during:** Task 1 — After copying look2hear/, the import failed with `ModuleNotFoundError: No module named 'vector_quantize_pytorch'` and then `ModuleNotFoundError: No module named 'taylor_series_linear_attention'`
- **Issue:** `pip install -r requirements.txt` appeared to succeed but installed into the wrong venv (see deviation 1); additional run with correct python binary was needed
- **Fix:** Ran `python3 -m pip install vector-quantize-pytorch taylor_series_linear_attention beartype einops huggingface_hub safetensors` explicitly
- **Files modified:** None
- **Commit:** 11a9f1a

**3. [Rule 3 - Missing] espeak-ng system package not installable (sudo required)**

- **Found during:** Task 1 — `sudo apt install -y espeak-ng` requires terminal password input
- **Issue:** Claude Code cannot interactively provide sudo password
- **Fix:** Skipped espeak-ng install; noted that Kokoro will use fallback G2P backend without it. The `KPipeline` import test still passes. espeak-ng can be installed manually by user when needed.
- **Impact:** Kokoro phonemizer backend degraded until `sudo apt install espeak-ng` is run manually
- **Commit:** 11a9f1a (noted in plan, not blocking)

## Auth Gates

None encountered.

## Verification Results

```
from look2hear.models import Dolphin  -> OK (CPU, no weights)
from kokoro import KPipeline          -> OK
from l2cs import Pipeline             -> OK
pytest --version                      -> pytest 9.0.2

pytest tests/unit/ -v
  15 passed, 5 xfailed, 2 warnings in 4.47s
```

## Self-Check

### Files Created/Modified

- [x] `look2hear/models/__init__.py` — exists
- [x] `look2hear/models/dolphin.py` — exists
- [x] `tests/conftest.py` — exists (32 lines, > 20 minimum)
- [x] `tests/unit/test_dolphin_separator.py` — exists (60 lines, > 15 minimum)
- [x] `tests/unit/test_tts.py` — exists (62 lines, > 15 minimum)
- [x] `tests/unit/test_gaze.py` — exists (76 lines, > 15 minimum)
- [x] `tests/unit/test_eou_detector.py` — exists (75 lines, > 15 minimum)
- [x] `smait/core/events.py` — modified (stale loop removed)
- [x] `pyproject.toml` — modified (pytest-cov added)
- [x] `.gitignore` — modified (vendor/ added)

### Commits

- [x] 11a9f1a — Task 1: vendor Dolphin, install deps
- [x] 1d3e9a1 — Task 2: test infrastructure, EventBus fix

### Must-Haves Verified

- [x] `from look2hear.models import Dolphin` succeeds on CPU without GPU
- [x] pytest discovers and runs tests from tests/ directory
- [x] EventBus does not cache stale asyncio loop between test runs

## Self-Check: PASSED
