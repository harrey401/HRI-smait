# Phase 1: Dependency Setup & Stub API Fixes - Research

**Researched:** 2026-03-09
**Domain:** Python dependency vendoring, ML library installation, stub API correction
**Confidence:** HIGH (all API errors verified against upstream source; Dolphin API confirmed from HuggingFace Space Inference.py)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-03 | Dolphin AV-TSE vendored from source (not pip — no setup.py exists) | Confirmed: no setup.py/pyproject.toml in JusperLee/Dolphin repo; vendoring look2hear/ is the correct install path |
| QUAL-01 | All stub APIs corrected to match real model interfaces | All four stub files have verified wrong imports; correct APIs documented with sources |
</phase_requirements>

---

## Summary

Phase 1 has a clear, bounded scope: fix four stub files that use fabricated ML APIs, vendor the Dolphin source package that cannot be pip-installed, install two packages (Kokoro, L2CS-Net) that can be, and stand up the test infrastructure from scratch. The project currently has zero automated tests — the `tests/` directory does not exist despite being listed in `pyproject.toml`.

All four stub errors are HIGH confidence because they were cross-checked against the actual upstream source code (Dolphin's HuggingFace Space `Inference.py`, Kokoro's PyPI package, L2CS-Net's GitHub). The errors are not subtle — `from dolphin import DolphinModel` and `from kokoro import KokoroTTS` are import paths that simply do not exist. The correct classes are `look2hear.models.Dolphin` and `kokoro.KPipeline` respectively.

The `eou_detector.py` stub is unique: it tries to load `livekit.plugins.turn_detector.EOUModel`, a private package that is unavailable. For Phase 1, the correct fix is to remove the LiveKit import attempt and leave the heuristic-only code path in place. The full VAD-based EOU rewrite is Phase 5 work.

**Primary recommendation:** Vendor `look2hear/` by cloning JusperLee/Dolphin into a `vendor/` directory and symlinking (or copying) the `look2hear/` subdirectory into the project root so `from look2hear.models import Dolphin` resolves without PYTHONPATH manipulation.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| kokoro | >=0.9.4 | Kokoro-82M TTS pipeline | pip-installable, KPipeline generator API, 24kHz output |
| espeak-ng | system package | Kokoro phonemizer backend | Required for G2P; Kokoro falls back to worse G2P without it |
| l2cs (edavalosanaya fork) | HEAD@main | Gaze estimation Pipeline | pip-installable fork; original Ahmednull repo lacks pip packaging |
| pytest | >=7.0.0 | Test runner | Already in pyproject.toml dev deps |
| pytest-asyncio | >=0.21.0 | Async test support | Already in pyproject.toml dev deps |
| pytest-cov | >=4.0.0 | Coverage measurement | Missing from dev deps — must add |

### Dolphin (Vendored, Not pip)

| Component | Source | Install method |
|-----------|--------|---------------|
| JusperLee/Dolphin repo | github.com/JusperLee/Dolphin | `git clone` into `vendor/dolphin-src/` |
| look2hear/ package | Subdirectory of Dolphin repo | Copy or symlink to project root |
| Dolphin dependencies | vendor/dolphin-src/requirements.txt | `pip install -r vendor/dolphin-src/requirements.txt` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Copying look2hear/ into project | PYTHONPATH export | Copying is more portable; PYTHONPATH approach requires env setup in every shell/service |
| edavalosanaya L2CS-Net fork | Original Ahmednull/L2CS-Net | Original lacks pip packaging; fork is maintained and pip-installable |

### Installation

```bash
# Kokoro TTS
pip install kokoro>=0.9.4 soundfile
sudo apt install espeak-ng

# L2CS-Net (maintained fork)
pip install "git+https://github.com/edavalosanaya/L2CS-Net.git@main"

# Dolphin (NOT pip installable)
mkdir -p vendor
git clone https://github.com/JusperLee/Dolphin.git vendor/dolphin-src
pip install -r vendor/dolphin-src/requirements.txt
# Then copy look2hear into project root (one-time):
cp -r vendor/dolphin-src/look2hear ./look2hear

# Test dependencies (add pytest-cov to pyproject.toml first)
pip install pytest pytest-asyncio pytest-cov
```

---

## Architecture Patterns

### Recommended Project Structure

```
smait-v3/
├── look2hear/           # Vendored from JusperLee/Dolphin (copy, not symlink for git cleanliness)
├── vendor/
│   └── dolphin-src/     # Full cloned Dolphin repo (for requirements.txt and reference)
├── smait/
│   ├── perception/
│   │   ├── dolphin_separator.py   # REWRITE with look2hear API
│   │   ├── gaze.py                # FIX arch param
│   │   └── eou_detector.py        # REMOVE LiveKit import, keep heuristic
│   └── output/
│       └── tts.py                 # REWRITE with KPipeline API
└── tests/
    ├── conftest.py
    └── unit/
        ├── test_dolphin_separator.py
        ├── test_tts.py
        ├── test_gaze.py
        └── test_eou_detector.py
```

### Pattern 1: Stub API Correction (All Four Files)

**What:** Replace fabricated imports with verified real imports. Keep the `_available` flag pattern and graceful fallback — that architecture is correct and reusable.

**When to use:** Every stub file in this phase.

**Dolphin correction:**
```python
# Source: HuggingFace Space Inference.py (huggingface.co/spaces/JusperLee/Dolphin)
# BEFORE (wrong):
from dolphin import DolphinModel
self._model = DolphinModel.from_pretrained("JusperLee/Dolphin")

# AFTER (correct):
from look2hear.models import Dolphin
self._model = Dolphin.from_pretrained("JusperLee/Dolphin")
self._model = self._model.to(self._device)
self._model.eval()
```

**Kokoro correction:**
```python
# Source: pypi.org/project/kokoro v0.9.4
# BEFORE (wrong):
from kokoro import KokoroTTS
self._model = KokoroTTS()

# AFTER (correct):
from kokoro import KPipeline
self._pipeline = KPipeline(lang_code='a')  # 'a' = American English
```

**L2CS-Net correction:**
```python
# Source: github.com/edavalosanaya/L2CS-Net
# BEFORE (wrong):
self._l2cs_pipeline = L2CSPipeline(weights=None, arch="Gaze360", device=self._device)

# AFTER (correct):
from l2cs import Pipeline as L2CSPipeline
self._l2cs_pipeline = L2CSPipeline(weights=None, arch='ResNet50', device=self._device)
```

**EOU Detector correction (Phase 1 scope only):**
```python
# BEFORE (wrong — both import paths fail):
from livekit.plugins.turn_detector import EOUModel
# fallback: from transformers import AutoModelForSequenceClassification
# AutoTokenizer.from_pretrained("livekit/turn-detector")  # private model

# AFTER (Phase 1 fix — remove model loading, rely on heuristic only):
# No ML model import. self._available = False. _heuristic_eou() is the sole predict() path.
# Full VAD-based rewrite is Phase 5.
```

### Pattern 2: Vendoring look2hear

**What:** Copy `look2hear/` from the cloned Dolphin repo into the project root so Python import resolution finds it without PYTHONPATH mutation.

**When to use:** One-time setup during Wave 1 of this phase.

**Why copy, not symlink:** Git tracks copies. Symlinks to locations outside the repo break on other machines. The look2hear package is ~2MB of Python — copying is fine.

**Verification:**
```python
# CPU-only import test (no GPU needed, no weights downloaded)
from look2hear.models import Dolphin
print("Dolphin import OK")
```

### Pattern 3: Test Infrastructure Bootstrap

**What:** Create `tests/` directory, `conftest.py`, and stub test files before writing any implementation. Tests use `unittest.mock` to stub out model `init_model()` calls — never call the real loaders in unit tests.

**Mock pattern for model stubs:**
```python
# For testing DolphinSeparator without GPU/weights:
from unittest.mock import MagicMock, patch
import pytest
from smait.perception.dolphin_separator import DolphinSeparator
from smait.core.config import Config
from smait.core.events import EventBus

@pytest.fixture
def separator():
    config = Config()
    bus = EventBus()
    sep = DolphinSeparator(config, bus)
    # Do NOT call init_model() — _available stays False, passthrough path tested
    return sep

def test_passthrough_returns_mono(separator):
    import numpy as np
    audio = np.zeros(16000, dtype=np.int16)
    import asyncio
    result = asyncio.run(separator.separate(audio, lip_frames=[], channels=1))
    assert result.separated_audio.dtype == np.float32
    assert result.separation_confidence == 0.0
```

### Anti-Patterns to Avoid

- **Calling `init_model()` in unit tests:** Downloads weights (GBs) and requires GPU. Use `_available=False` fallback paths or mock the `_model` attribute directly.
- **Importing `DolphinModel`, `KokoroTTS`, or `EOUModel` anywhere:** These classes do not exist. If they appear in any import, the file has not been fixed.
- **Using PYTHONPATH for look2hear:** Fragile across environments. Copy look2hear/ into the project root instead.
- **Installing from original Ahmednull/L2CS-Net:** That repo may lack pip support. Use `edavalosanaya/L2CS-Net` fork.
- **Rewriting eou_detector.py fully in Phase 1:** Phase 1 scope is only removing the broken LiveKit import. Full VAD-based EOU rewrite is Phase 5. Leave heuristic logic intact.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentence boundary detection | Custom regex EOU detector | Kokoro KPipeline already yields per-sentence audio | KPipeline splits internally; double-splitting wastes latency |
| Gaze angle estimation | Head pose math | l2cs.Pipeline | L2CS-Net ResNet50 trained on Gaze360 dataset; hand-rolled head pose is inaccurate |
| Model weight downloading | Custom HuggingFace downloader | `Dolphin.from_pretrained("JusperLee/Dolphin")` | Uses huggingface_hub with caching, retry, and auth |
| Test async code | Manual `asyncio.run()` wrappers | pytest-asyncio with `asyncio_mode = "auto"` | Already configured in pyproject.toml |

**Key insight:** The stub architecture (graceful fallback, `_available` flags, try/except around imports) is correct and well-designed. The only problem is the import paths inside the try blocks. Fix those — don't refactor the surrounding structure.

---

## Common Pitfalls

### Pitfall 1: Dolphin `from dolphin import DolphinModel` — Package Does Not Exist

**What goes wrong:** The stub tries to import from a `dolphin` top-level package. No such package exists anywhere (not on PyPI, not as a module in the repo). Even if you clone the Dolphin repo and add it to PYTHONPATH, the importable package is `look2hear`, not `dolphin`.

**Why it happens:** Stub was written speculatively without checking the repo.

**How to avoid:** Use `from look2hear.models import Dolphin`. The `look2hear/` directory is inside the cloned Dolphin repo at `vendor/dolphin-src/look2hear/`.

**Warning signs:** `ModuleNotFoundError: No module named 'dolphin'` on any import attempt.

### Pitfall 2: Kokoro `KokoroTTS` — Class Does Not Exist

**What goes wrong:** `from kokoro import KokoroTTS` raises `ImportError: cannot import name 'KokoroTTS'` even after `pip install kokoro`. The class is not in the package.

**How to avoid:** Use `from kokoro import KPipeline`. The pipeline is constructed as `KPipeline(lang_code='a')` and called as a generator: `for gs, ps, audio in pipeline(text, voice='af_heart')`.

**Warning signs:** `ImportError: cannot import name 'KokoroTTS' from 'kokoro'`.

### Pitfall 3: L2CS-Net `arch="Gaze360"` Is Wrong

**What goes wrong:** The stub passes `arch="Gaze360"` to `L2CSPipeline`. The Pipeline API's `arch` parameter expects the backbone architecture name, not the dataset name. The correct value is `'ResNet50'`.

**How to avoid:** Use `arch='ResNet50'`. The weights are trained on Gaze360 data but the architecture name is ResNet50.

**Warning signs:** Runtime error or wrong initialization inside L2CS-Net.

### Pitfall 4: LiveKit EOU Model Is Completely Unavailable

**What goes wrong:** Both `livekit.plugins.turn_detector` (private pip package) and `livekit/turn-detector` on HuggingFace (private model repo) are inaccessible. The fallback in the stub (`AutoModelForSequenceClassification.from_pretrained("livekit/turn-detector")`) also fails with a 404/access-denied error.

**How to avoid (Phase 1):** Remove the `init_model()` body entirely or replace it with a log message. Keep `self._available = False`. The heuristic `_heuristic_eou()` method is the correct code path for Phase 1. Phase 5 will replace the whole class with a VAD-silence approach.

**Warning signs:** `OSError: livekit/turn-detector is not a local folder and is not a valid model identifier on HuggingFace Hub`.

### Pitfall 5: Dolphin Tensor Shape Mismatches

**What goes wrong:** The stub passes audio as `(batch, channels, samples)` and video as `(batch, frames, H, W, C)`. Dolphin requires `(batch, samples)` mono audio and `(batch, 1, frames, 88, 88, 1)` grayscale video. Shape mismatches produce cryptic CUDA/PyTorch errors or silent wrong output.

**How to avoid:** In Phase 1, the DolphinSeparator rewrite should define the correct tensor prep even though the model won't be loaded yet. The `_run_dolphin()` method should prepare tensors in the correct shape so Phase 4 can wire it to real weights without re-investigation.

**Warning signs:** `RuntimeError: Expected tensor with 2 dimensions` or similar shape mismatch errors when eventually running with real weights in Phase 7.

### Pitfall 6: EventBus Stale Loop in Tests

**What goes wrong:** `EventBus._loop` caches the asyncio loop on first `emit()`. If tests call `asyncio.run()` multiple times (each creates a new loop), the cached loop is invalid. Async handlers silently never fire.

**How to avoid:** Fix EventBus to call `asyncio.get_running_loop()` on each emit rather than caching. Add this fix in Phase 1 since it will affect all subsequent test work.

**Warning signs:** Async test handlers never fire; captured event lists stay empty.

### Pitfall 7: No tests/ Directory

**What goes wrong:** `pyproject.toml` specifies `testpaths = ["tests"]` but the directory doesn't exist. Running `pytest` exits immediately with "no tests ran" rather than an error, making it easy to think tests passed.

**How to avoid:** Create `tests/conftest.py` and at least one real test file in Wave 0 of this phase before writing any implementation code.

**Warning signs:** `pytest` reports `no tests ran` or `collected 0 items`.

---

## Code Examples

### Correct Dolphin Initialization

```python
# Source: huggingface.co/spaces/JusperLee/Dolphin (Inference.py, verified 2026-03-09)
async def init_model(self) -> None:
    logger.info("Loading Dolphin AV-TSE model...")
    try:
        from look2hear.models import Dolphin
        self._model = Dolphin.from_pretrained("JusperLee/Dolphin")
        self._model = self._model.to(self._device)
        self._model.eval()
        self._available = True
        logger.info("Dolphin AV-TSE loaded on %s", self._device)
    except ImportError:
        logger.warning(
            "look2hear not found. Vendor Dolphin: "
            "git clone https://github.com/JusperLee/Dolphin.git vendor/dolphin-src "
            "and copy vendor/dolphin-src/look2hear into the project root."
        )
    except Exception:
        logger.exception("Failed to load Dolphin model")
```

### Correct Dolphin Tensor Shapes

```python
# Source: huggingface.co/spaces/JusperLee/Dolphin (Inference.py, verified 2026-03-09)
# Audio input: [1, num_samples] mono float32 at 16kHz
audio_tensor = torch.from_numpy(mono_float32_audio).unsqueeze(0).to(self._device)
# Shape: [1, samples]

# Video input: [1, 1, num_frames, 88, 88, 1] grayscale uint8 or float32
# lip_stack shape: [T, 88, 88, 1]
lip_stack = np.stack([roi for roi in gray_88x88_rois], axis=0)[..., np.newaxis]
video_tensor = torch.from_numpy(lip_stack[np.newaxis, np.newaxis]).float().to(self._device)
# Shape: [1, 1, T, 88, 88, 1]

with torch.no_grad():
    est_sources = self._model(audio_tensor, video_tensor)
# Output: [1, num_samples] separated audio
```

### Correct Kokoro Initialization and Generation

```python
# Source: pypi.org/project/kokoro v0.9.4 + huggingface.co/hexgrad/Kokoro-82M
async def init_model(self) -> None:
    logger.info("Loading Kokoro-82M TTS model...")
    try:
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code='a')  # 'a' = American English
        self._voice = getattr(self._config, 'voice', 'af_heart')
        self._available = True
        logger.info("Kokoro-82M loaded (voice=%s, sample_rate=%d)", self._voice, self._sample_rate)
    except ImportError:
        logger.warning(
            "Kokoro TTS not installed. Run: pip install kokoro>=0.9.4 soundfile && sudo apt install espeak-ng"
        )
    except Exception:
        logger.exception("Failed to load Kokoro model")

async def synthesize(self, text: str) -> Optional[bytes]:
    if not self._available or self._pipeline is None:
        return None
    try:
        pcm_parts = []
        for _graphemes, _phonemes, audio in self._pipeline(text, voice=self._voice, speed=1.0):
            # audio: numpy float32 at 24kHz
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_parts.append(pcm.tobytes())
        return b"".join(pcm_parts)
    except Exception:
        logger.exception("TTS synthesis failed")
        return None
```

### Correct L2CS-Net Initialization

```python
# Source: github.com/edavalosanaya/L2CS-Net (verified 2026-03-09)
async def init_model(self) -> None:
    logger.info("Loading L2CS-Net gaze model...")
    try:
        from l2cs import Pipeline as L2CSPipeline
        self._l2cs_pipeline = L2CSPipeline(
            weights=None,          # auto-download (or set to local path)
            arch='ResNet50',       # NOT 'Gaze360'
            device=self._device,
        )
        logger.info("L2CS-Net loaded on %s", self._device)
    except ImportError:
        logger.warning(
            "L2CS-Net not installed. Run: "
            "pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main"
        )
        self._l2cs_pipeline = None
```

### Correct EOU Detector (Phase 1 — heuristic only, no LiveKit)

```python
# Phase 1: remove all LiveKit model loading. _available = False. Heuristic path only.
async def init_model(self) -> None:
    """EOU model loading deferred to Phase 5 (VAD-based rewrite).
    Phase 1: mark as unavailable; heuristic predict() path is active.
    """
    logger.info(
        "EOUDetector: using heuristic fallback (VAD-based EOU rewrite in Phase 5). "
        "LiveKit turn-detector is unavailable."
    )
    self._available = False
```

### Test Infrastructure (conftest.py)

```python
# tests/conftest.py
import pytest
import numpy as np
from smait.core.config import Config, reset_config
from smait.core.events import EventBus


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset config singleton between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def silence_audio():
    """16kHz mono silence, 1 second."""
    return np.zeros(16000, dtype=np.int16)


@pytest.fixture
def speech_audio():
    """16kHz mono noise simulating speech, 1 second."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(16000) * 3000).astype(np.int16)
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `from dolphin import DolphinModel` | `from look2hear.models import Dolphin` | ImportError -> working import |
| `DolphinModel.from_pretrained(...)` | `Dolphin.from_pretrained("JusperLee/Dolphin")` | AttributeError -> correct load |
| Audio: `(batch, channels, samples)` | Audio: `(batch, samples)` mono | Shape mismatch -> correct inference |
| Video: `(batch, frames, H, W, C)` | Video: `(batch, 1, frames, 88, 88, 1)` | Shape mismatch -> correct inference |
| `from kokoro import KokoroTTS` | `from kokoro import KPipeline` | ImportError -> working import |
| `model.generate(text)` | `pipeline(text, voice='af_heart')` generator | Wrong API -> streaming generator |
| `arch="Gaze360"` | `arch='ResNet50'` | Wrong init param -> correct backbone |
| LiveKit EOU model import | Heuristic only (Phase 1); VAD thresholds (Phase 5) | Unavailable -> working fallback |

**Deprecated/outdated:**
- `dolphin` package name: never existed; replaced by `look2hear`
- `KokoroTTS` class: never existed; replaced by `KPipeline`
- `EOUModel` from livekit.plugins: private, unavailable; replaced by heuristic + VAD silence

---

## Open Questions

1. **look2hear preprocessing pipeline**
   - What we know: Dolphin's `Inference.py` calls `get_preprocessing_pipelines()["val"]` for mouth ROI normalization
   - What's unclear: Whether raw 88x88 grayscale float32 (without the preprocessing pipeline) is sufficient, or if specific normalization is required
   - Recommendation: In Phase 1, document the question and leave a TODO comment in `_preprocess_lip_frames()`. Phase 4 will investigate by running real inference.

2. **Dolphin dependency conflicts with existing stack**
   - What we know: Dolphin requirements include `face_alignment`, `retina-face`, `tf-keras` — packages with complex dependencies
   - What's unclear: Whether these conflict with MediaPipe or other existing packages
   - Recommendation: Install Dolphin requirements in a fresh venv to detect conflicts before installing into the project venv.

3. **L2CS-Net weight auto-download reliability**
   - What we know: Weights hosted on Google Drive, known rate limits
   - What's unclear: Whether the maintained fork (edavalosanaya) hosts weights differently
   - Recommendation: During Phase 1 setup, attempt auto-download once and note the local cache path. Pre-download to a known path and set `weights=` explicitly for offline/lab use.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.0.0+ with pytest-asyncio 0.21.0+ |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `pytest tests/unit/ -x -q` |
| Full suite command | `pytest tests/ --cov=smait --cov-report=term-missing` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-03 | `from look2hear.models import Dolphin` succeeds (CPU, no weights) | unit (import test) | `pytest tests/unit/test_dolphin_separator.py::test_look2hear_importable -x` | Wave 0 |
| ENV-03 | DolphinSeparator passthrough returns mono float32 when unavailable | unit | `pytest tests/unit/test_dolphin_separator.py::test_passthrough_returns_mono -x` | Wave 0 |
| QUAL-01 | TTSEngine init_model() calls KPipeline not KokoroTTS | unit (mock) | `pytest tests/unit/test_tts.py::test_correct_class_imported -x` | Wave 0 |
| QUAL-01 | GazeEstimator uses arch='ResNet50' not 'Gaze360' | unit (mock) | `pytest tests/unit/test_gaze.py::test_correct_arch_param -x` | Wave 0 |
| QUAL-01 | EOUDetector.predict() returns float from heuristic when unavailable | unit | `pytest tests/unit/test_eou_detector.py::test_heuristic_predict -x` | Wave 0 |
| QUAL-01 | EOUDetector never attempts LiveKit import | unit (mock) | `pytest tests/unit/test_eou_detector.py::test_no_livekit_import -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/unit/ -x -q`
- **Per wave merge:** `pytest tests/ --cov=smait --cov-report=term-missing`
- **Phase gate:** All unit tests green, `from look2hear.models import Dolphin` import test passes

### Wave 0 Gaps

- [ ] `tests/conftest.py` — shared fixtures (config, event_bus, silence_audio, speech_audio)
- [ ] `tests/unit/test_dolphin_separator.py` — covers ENV-03, QUAL-01 (dolphin)
- [ ] `tests/unit/test_tts.py` — covers QUAL-01 (kokoro)
- [ ] `tests/unit/test_gaze.py` — covers QUAL-01 (l2cs)
- [ ] `tests/unit/test_eou_detector.py` — covers QUAL-01 (eou)
- [ ] Framework install: `pip install pytest-cov` — missing from dev deps; add to `pyproject.toml [project.optional-dependencies] dev`

---

## Sources

### Primary (HIGH confidence)

- HuggingFace Space Inference.py (huggingface.co/spaces/JusperLee/Dolphin) — verified Dolphin import path, tensor shapes, forward call signature
- JusperLee/Dolphin HuggingFace (huggingface.co/JusperLee/Dolphin) — model weights location (conf.yml + best_model.pth)
- Kokoro PyPI v0.9.4 (pypi.org/project/kokoro) — KPipeline API, lang_code, voice list
- hexgrad/Kokoro-82M HuggingFace (huggingface.co/hexgrad/Kokoro-82M) — usage examples, voice list
- Project's own `.planning/research/STACK.md` — compiled from above sources on 2026-03-09
- Project's own `.planning/research/PITFALLS.md` — stub error inventory

### Secondary (MEDIUM confidence)

- edavalosanaya/L2CS-Net GitHub (github.com/edavalosanaya/L2CS-Net) — pip-installable fork, Pipeline API with arch param
- .planning/codebase/TESTING.md — test infrastructure analysis, testable components inventory

### Tertiary (LOW confidence)

- None — all critical claims verified from primary sources

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library versions and install paths verified from PyPI or repo source
- Architecture: HIGH — all API corrections verified from upstream source code (Inference.py)
- Pitfalls: HIGH — import errors are deterministic; tensor shapes verified from upstream

**Research date:** 2026-03-09
**Valid until:** 2026-06-09 (stable APIs; Kokoro and l2cs change infrequently; Dolphin has no versioned releases)
