# Phase 7: GPU Validation & Model Loading - Research

**Researched:** 2026-03-11
**Domain:** CUDA 12.8 / Blackwell sm_120, PyTorch 2.7+, NeMo Parakeet TDT ASR, VRAM budgeting, GPU smoke-test patterns
**Confidence:** MEDIUM — PyTorch 2.7 + cu128 install commands verified via official blog; NeMo on Blackwell confirmed supported in principle but live lab behavior is inherently environment-specific

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | PyTorch nightly with CUDA 12.8+ verified on Blackwell sm_120 | PyTorch 2.7 cu128 install confirmed; `torch.cuda.get_arch_list()` shows sm_120; verification script pattern documented |
| ENV-02 | All ML models load simultaneously within 12GB VRAM budget | VRAM estimates per model documented; budget analysis ~8-10GB total headroom under 12GB |
| ASR-01 | Parakeet TDT ASR verified on Blackwell sm_120 with CUDA graphs disabled | `NEMO_DISABLE_CUDA_GRAPHS=1` env var already coded in asr.py; confirmed correct approach for Blackwell |
</phase_requirements>

---

## Summary

Phase 7 is the first LAB-only phase. Its job is to prove the RTX 5070 (Blackwell, sm_120) actually runs every model that was written and unit-tested at home. There is no new code to ship — the deliverable is a **validated environment script** and **evidence** (nvidia-smi output, model load logs, a real transcription).

The core risk is the PyTorch / CUDA 12.8 / NeMo compatibility stack. PyTorch 2.7 (released late 2025) is the first stable release that includes Blackwell sm_120 kernels. The install path is now straightforward on Linux, but NeMo has its own dependency constraints (numpy, Cython, specific torch ranges) that can conflict. The plan must sequence dependency installation carefully and verify at each layer before proceeding.

A secondary risk is VRAM. The budget is tight (12GB on RTX 5070, actual usable ~11.5GB after driver overhead). VRAM estimates below total ~8-10GB for all five models simultaneously, which leaves ~2GB headroom — safe, but not generous. Models must be loaded in the correct order (Parakeet first, then Dolphin, then Kokoro, then L2CS-Net, then Silero VAD) to catch OOM failures early against the heaviest consumers.

Parakeet TDT on Blackwell requires disabling CUDA graphs (`NEMO_DISABLE_CUDA_GRAPHS=1`). This is already coded in `smait/perception/asr.py` via `os.environ.setdefault(...)`. The validation plan must confirm this env var is set before NeMo loads.

**Primary recommendation:** Install PyTorch 2.7 + cu128 first, verify sm_120 with a standalone torch smoke test, then install NeMo independently, then run the SMAIT model loader. Gate each step — do not proceed to the next model if the previous one fails.

---

## Standard Stack

### Core (GPU Layer)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.7.0 cu128 | Deep learning runtime, CUDA execution | First stable release with Blackwell sm_120 kernels |
| torchaudio | 2.7.0 cu128 | Audio tensor ops used by VAD and ASR | Must match torch version exactly |
| CUDA Toolkit | 12.8 | GPU compute, cuDNN, NCCL | First CUDA version with native sm_120 support |
| NVIDIA Driver | R570+ | Hardware access | Required for CUDA 12.8 |

### ML Models

| Library | Version | Purpose | VRAM Estimate |
|---------|---------|---------|---------------|
| nemo_toolkit[asr] | 2.x (latest) | Parakeet TDT ASR wrapper | ~2GB (inference) |
| kokoro | >=0.9.4 | Kokoro-82M TTS | ~1GB (82M params, FP32) |
| look2hear (vendored) | from source | Dolphin AV-TSE separation | ~2-3GB (51.3M params, video encoder) |
| l2cs-net | from GitHub fork | Gaze estimation ResNet50 | ~0.5GB |
| silero-vad | via torch.hub | Voice activity detection | <0.1GB (tiny LSTM) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| nvidia-ml-py | any | Python bindings for nvidia-smi queries | VRAM budget verification |
| soundfile | >=0.12 | Load .wav test files for ASR smoke test | Audio fixture loading |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch 2.7 stable cu128 | Nightly cu128/cu129 | Nightly is newer but less predictable; stable 2.7 has confirmed sm_120 support |
| nvidia-ml-py | subprocess nvidia-smi | Python API is cleaner in scripts; subprocess is fine for quick checks |

**Installation (order matters):**

```bash
# Step 1: PyTorch + CUDA 12.8 (stable Blackwell build)
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Step 2: NeMo prerequisites (must come before nemo_toolkit)
pip install Cython packaging

# Step 3: NeMo ASR
pip install nemo_toolkit[asr]

# Step 4: Remaining project deps
pip install kokoro>=0.9.4
pip install git+https://github.com/Ahmednull/L2CS-Net.git
pip install nvidia-ml-py soundfile
```

**Note:** If `nemo_toolkit[asr]` downgrades torch to a pre-2.7 version, pin it back immediately:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

---

## Architecture Patterns

### Recommended Validation Script Structure

```
scripts/
├── validate_gpu.py        # master validation script (Phase 7)
├── smoke_torch.py         # standalone PyTorch + sm_120 check
├── smoke_parakeet.py      # Parakeet TDT + CUDA graphs disabled
├── smoke_all_models.py    # load all 5 models, report VRAM
└── test_audio/
    └── sample_16k.wav     # 3-5 second clean speech sample for ASR
```

### Pattern 1: Layered Smoke Test (Gate Each Step)

**What:** Run isolated verification for each layer before combining them.
**When to use:** Always for new GPU environments. Catching failures in isolation is faster than debugging a combined load.

```python
# smoke_torch.py — verify sm_120 before anything else
import torch

def verify_blackwell():
    assert torch.cuda.is_available(), "CUDA not available"
    arch_list = torch.cuda.get_arch_list()
    assert "sm_120" in arch_list, f"sm_120 not in arch list: {arch_list}"
    cap = torch.cuda.get_device_capability()
    assert cap >= (12, 0), f"Expected sm_120 capability, got {cap}"
    # Confirm a tensor op works on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    assert y.shape == (1000, 1000)
    print(f"PyTorch {torch.__version__} — sm_120 OK — capability {cap}")

if __name__ == "__main__":
    verify_blackwell()
```

### Pattern 2: VRAM Budget Check

**What:** Use `torch.cuda.memory_allocated()` and `nvidia-ml-py` to measure VRAM after each model loads.
**When to use:** After loading each model in the combined load test.

```python
# Source: PyTorch CUDA memory API (stable)
import torch

def vram_report(label: str) -> None:
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")

# Usage: call after each model load
vram_report("after Parakeet load")
vram_report("after Dolphin load")
# etc.
```

### Pattern 3: Parakeet with CUDA Graphs Disabled

**What:** Set `NEMO_DISABLE_CUDA_GRAPHS=1` before NeMo imports, then transcribe a real audio file.
**When to use:** This is ASR-01 validation.

```python
# smoke_parakeet.py
import os
os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"  # MUST be before NeMo import

import numpy as np
import soundfile as sf
import nemo.collections.asr as nemo_asr

def validate_parakeet():
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    model.eval()
    model.cuda()

    # Load real audio (16kHz mono float32)
    audio, sr = sf.read("scripts/test_audio/sample_16k.wav")
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    audio = audio.astype(np.float32)

    result = model.transcribe([audio], return_hypotheses=True)
    text = result[0][0].text if hasattr(result[0][0], 'text') else str(result[0][0])
    print(f"Parakeet transcription: '{text}'")
    assert len(text) > 0, "Empty transcription"
    print("ASR-01 PASS")

if __name__ == "__main__":
    validate_parakeet()
```

### Anti-Patterns to Avoid

- **Loading all models in one giant script without gate checks:** If Parakeet OOMs, it is unclear whether the issue is Parakeet itself or prior model footprint. Always load individually first.
- **Setting `NEMO_DISABLE_CUDA_GRAPHS=1` after `import nemo`:** The env var must be set before the first NeMo import or it has no effect on module-level graph compilation.
- **Assuming `torch.load` uses `weights_only=False`:** PyTorch 2.6+ defaults to `weights_only=True`. If model checkpoints fail to load, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (only for trusted checkpoints).
- **Running Dolphin with 4-channel audio on GPU:** Dolphin takes mono `[1, samples]`. The code in `main.py` already enforces this, but confirm the GPU path does not inadvertently pass `[4, samples]` when testing.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| VRAM measurement | Custom /proc/meminfo parser | `torch.cuda.memory_allocated()` + `nvidia-ml-py` | PyTorch's allocator tracks tensor-level usage; nvidia-ml-py adds driver-level total |
| Blackwell detection | String match on nvidia-smi output | `torch.cuda.get_device_capability()` returns `(major, minor)` tuple | Stable API, no parsing |
| Audio fixture | Sine wave generation | `soundfile` to load a real .wav | ASR must be tested with real speech, not synthetic audio |
| Environment validation | Assert in production code | Standalone `validate_gpu.py` script | Keep validation separate from the runtime path |

**Key insight:** PyTorch already exposes everything needed to verify GPU capability. The only custom code is sequencing and reporting.

---

## Common Pitfalls

### Pitfall 1: NeMo Downgrades PyTorch

**What goes wrong:** `pip install nemo_toolkit[asr]` resolves torch to an older version (e.g., 2.5.x) that does not include sm_120 kernels. Imports succeed but GPU tests silently use CPU fallback.
**Why it happens:** NeMo's dependency constraints may pin torch to a range that predates 2.7.
**How to avoid:** After installing NeMo, immediately verify `torch.__version__` is 2.7.0. If downgraded, force-reinstall PyTorch 2.7 cu128 and retest.
**Warning signs:** `torch.cuda.get_arch_list()` no longer contains sm_120 after NeMo install.

### Pitfall 2: NEMO_DISABLE_CUDA_GRAPHS Set Too Late

**What goes wrong:** `NEMO_DISABLE_CUDA_GRAPHS=1` is set after `import nemo` and CUDA graphs compile on Blackwell, causing a crash or silent hang during the first transcribe call.
**Why it happens:** NeMo checks this env var during module-level imports, not at call time.
**How to avoid:** Set env var as the very first line of any script that uses NeMo, before all imports.
**Warning signs:** First `model.transcribe()` hangs > 60 seconds or produces a CUDA assertion error.

### Pitfall 3: VRAM Budget Exceeded by Footprint Growth

**What goes wrong:** Models load individually without OOM, but loading all five simultaneously causes an OOM error.
**Why it happens:** PyTorch's memory allocator caches freed memory (reserved > allocated). Activations from model loading can leave fragmented reserved blocks.
**How to avoid:** After each model load verification, call `torch.cuda.empty_cache()` before loading the next model in the combined test. Measure after cache clear.
**Warning signs:** `torch.cuda.memory_reserved()` exceeds 11GB even if `memory_allocated()` looks fine.

### Pitfall 4: Dolphin Video Tensor Wrong Shape on GPU

**What goes wrong:** Dolphin GPU inference fails with shape mismatch because video tensor is `[1, frames, 1, 88, 88]` instead of the expected `[1, 1, frames, 88, 88, 1]`.
**Why it happens:** The CPU unit tests used mocked Dolphin. Real GPU inference exposes the actual model's expected tensor layout.
**How to avoid:** Confirm the exact input shape using the Dolphin HuggingFace model card: `audio_mixture = torch.randn(1, 64000)` and `video_frames = torch.randn(1, 100, 1, 88, 88)` for 4-second clip at 25fps. Cross-check against `DolphinSeparator._run_dolphin()` tensor construction.
**Warning signs:** `RuntimeError: Expected 5D tensor` or similar during first real GPU separation call.

### Pitfall 5: `torch.load` Weights-Only Default

**What goes wrong:** NeMo or Dolphin checkpoints fail to load with `_pickle.UnpicklingError` or a weights_only error on PyTorch 2.6+.
**Why it happens:** PyTorch 2.6+ changed `torch.load` default to `weights_only=True`.
**How to avoid:** If any model fails to load checkpoints, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` environment variable before running (trusted checkpoints only).
**Warning signs:** `RuntimeError: Weights only load failed` in model loading traceback.

---

## Code Examples

### Verify sm_120 Capability

```python
# Source: PyTorch CUDA API (stable in 2.7)
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))  # expects (12, 0) for sm_120
print(torch.cuda.get_arch_list())           # expects 'sm_120' in list
```

### Load Parakeet with CUDA Graphs Disabled

```python
# Source: smait/perception/asr.py (existing production code)
import os
os.environ.setdefault("NEMO_DISABLE_CUDA_GRAPHS", "1")  # before NeMo import

import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
model.eval()
```

### VRAM Budget Audit

```python
# Source: PyTorch CUDA memory docs + nvidia-ml-py
import torch
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def full_vram_report(label: str) -> None:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    torch_alloc = torch.cuda.memory_allocated() / 1024**3
    torch_reserved = torch.cuda.memory_reserved() / 1024**3
    driver_used = info.used / 1024**3
    driver_total = info.total / 1024**3
    print(
        f"[VRAM:{label}] "
        f"torch_alloc={torch_alloc:.2f}GB "
        f"torch_reserved={torch_reserved:.2f}GB "
        f"driver_used={driver_used:.2f}GB / {driver_total:.2f}GB total"
    )
```

### Combined Model Load Test (Sequential with Gates)

```python
# Pattern: load each model, report VRAM, assert under budget
VRAM_BUDGET_GB = 11.5  # 12GB card minus ~0.5GB driver overhead

models = {}

# 1. Parakeet (~2GB)
models["parakeet"] = load_parakeet()
full_vram_report("after parakeet")

# 2. Dolphin (~2-3GB)
models["dolphin"] = load_dolphin()
full_vram_report("after dolphin")

# 3. Kokoro (~1GB)
models["kokoro"] = load_kokoro()
full_vram_report("after kokoro")

# 4. L2CS-Net (~0.5GB)
models["l2cs"] = load_l2cs()
full_vram_report("after l2cs")

# 5. Silero VAD (<0.1GB)
models["vad"] = load_silero_vad()
full_vram_report("after silero_vad")

# Final budget check
driver_used = get_driver_vram_gb()
assert driver_used < VRAM_BUDGET_GB, f"VRAM over budget: {driver_used:.2f}GB > {VRAM_BUDGET_GB}GB"
print("ENV-02 PASS")
```

---

## VRAM Budget Analysis

This is a best-estimate based on available model documentation. Actual usage must be measured in lab.

| Model | Estimated VRAM | Source / Confidence |
|-------|---------------|---------------------|
| Parakeet TDT 0.6B | ~2 GB | HuggingFace card: "2GB minimum RAM for model load" — MEDIUM |
| Dolphin AV-TSE | ~2-3 GB | 51.3M params; video encoder 8.5M; MACs 417G — LOW (no official VRAM spec) |
| Kokoro-82M TTS | ~1 GB | 82M params FP32; community reports 2-4GB for FP32; ~1GB FP16 — MEDIUM |
| L2CS-Net ResNet50 | ~0.5 GB | ResNet50 backbone ~25M params; standard ResNet50 ~0.5GB — MEDIUM |
| Silero VAD | <0.1 GB | Tiny LSTM, runs on CPU in production; loaded via torch.hub — HIGH |
| **Total estimate** | **~5.6-6.6 GB** | Low-end estimate; peak during inference may be higher |
| **Peak inference headroom** | **~3-4 GB** | Activations during simultaneous forward pass |
| **Budget (RTX 5070)** | **12 GB** | Confirmed hardware |

**Verdict:** Budget should be comfortable under normal conditions. The main risk is Dolphin's inference-time activation footprint, which is not documented. Measure in lab.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PyTorch nightly cu128 for Blackwell | PyTorch 2.7 stable cu128 | Late 2025 | Can now use stable builds; nightly no longer required |
| Manually disable CUDA graphs via config | `NEMO_DISABLE_CUDA_GRAPHS=1` env var | NeMo 2.x | Single env var replaces complex model config surgery |
| `torch.load()` accepts any pickle | `torch.load()` defaults `weights_only=True` | PyTorch 2.6 | Checkpoint loading may need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |

**Deprecated/outdated:**
- `--index-url https://download.pytorch.org/whl/nightly/cu128`: Was the only option before PyTorch 2.7. Now use stable cu128.
- CUDA Toolkit 12.4 and earlier: sm_120 not defined. Must be 12.8+.

---

## Open Questions

1. **Does NeMo's latest pip release pin torch below 2.7?**
   - What we know: NeMo 2.x installs via `pip install nemo_toolkit[asr]`; there was a known numpy 2.0 conflict as of mid-2025
   - What's unclear: Whether the current release (as of March 2026) constrains torch to < 2.7 or tolerates 2.7
   - Recommendation: After NeMo install, immediately check `pip show torch` and force-reinstall 2.7 if needed

2. **What is Dolphin's actual GPU VRAM footprint during inference?**
   - What we know: 51.3M parameters; can run on CPU; designed for edge devices
   - What's unclear: Peak VRAM during `model(audio, video)` forward pass with real-sized tensors
   - Recommendation: Profile with `torch.cuda.max_memory_allocated()` before and after first inference call

3. **Does Parakeet's `from_pretrained()` download weights at each run or cache them?**
   - What we know: NeMo models are downloaded from HuggingFace/NGC on first call
   - What's unclear: Cache directory on the lab machine; network access from lab
   - Recommendation: First run needs internet access; thereafter `~/.cache/huggingface/hub` is used. Confirm lab has outbound HTTPS.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 7.x with pytest-asyncio |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]` asyncio_mode = "auto") |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ --cov=smait --cov-report=term-missing` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | PyTorch loads and reports sm_120 on RTX 5070 | smoke (GPU) | `python scripts/smoke_torch.py` | ❌ Wave 0 |
| ENV-02 | All 5 models load simultaneously under 12GB VRAM | smoke (GPU) | `python scripts/smoke_all_models.py` | ❌ Wave 0 |
| ASR-01 | Parakeet transcribes real audio on GPU with CUDA graphs disabled | smoke (GPU) | `python scripts/smoke_parakeet.py` | ❌ Wave 0 |

**Note:** Phase 7 smoke tests are GPU-executable scripts (not pytest unit tests) because they require the physical RTX 5070. They cannot be run in the home CI environment. They live in `scripts/` and are executed manually in lab. The existing unit test suite continues to run without GPU.

### Sampling Rate

- **Per task commit:** `pytest tests/ -x -q` (unit tests, no GPU required)
- **Per wave merge:** `pytest tests/ --cov=smait --cov-report=term-missing`
- **Phase gate (GPU):** All three smoke scripts pass on RTX 5070 before declaring Phase 7 complete

### Wave 0 Gaps

- [ ] `scripts/smoke_torch.py` — ENV-01: PyTorch sm_120 verification
- [ ] `scripts/smoke_parakeet.py` — ASR-01: Parakeet TDT GPU transcription
- [ ] `scripts/smoke_all_models.py` — ENV-02: simultaneous VRAM budget check
- [ ] `scripts/test_audio/sample_16k.wav` — 3-5 second real English speech for ASR smoke test
- [ ] `scripts/validate_gpu.py` — master script that runs all three smoke tests in sequence

---

## Sources

### Primary (HIGH confidence)

- PyTorch 2.7 Release Blog — https://pytorch.org/blog/pytorch-2-7/ — Confirmed cu128 stable build, Blackwell support, Triton 3.3
- nvidia/parakeet-tdt-0.6b-v2 HuggingFace — https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 — 2GB VRAM minimum, Blackwell listed as supported architecture
- JusperLee/Dolphin HuggingFace — https://huggingface.co/JusperLee/Dolphin — 51.3M params, inference example code

### Secondary (MEDIUM confidence)

- NVIDIA Developer Forums Blackwell Migration Guide — https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330 — CUDA 12.8 mandatory for sm_120, R570+ driver
- PyTorch Forums sm_120 thread — https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099 — Community-verified arch_list including sm_120 after PyTorch 2.7 cu128 install
- NeMo Issue #15164 (CUDA graphs) — https://github.com/NVIDIA-NeMo/NeMo/issues/15164 — CUDA graph behavior in TDT decoder, confirmed `torch.bfloat16` + `inference_mode` pattern
- pytorch/pytorch Issue #164342 — https://github.com/pytorch/pytorch/issues/164342 — sm_120 added to stable builds as of Oct 2025

### Tertiary (LOW confidence)

- Kokoro VRAM "2-4GB" estimate: derived from 82M parameter count and community reports (not official documentation)
- L2CS-Net VRAM "~0.5GB" estimate: derived from ResNet50 backbone size (25M params) — no official VRAM spec found
- Silero VAD "<0.1GB": inference from CPU-optimized design; no GPU VRAM measurement found in official docs

---

## Metadata

**Confidence breakdown:**

- PyTorch 2.7 cu128 install: HIGH — verified via official release blog and PyTorch forums
- sm_120 verification commands: HIGH — stable PyTorch CUDA API
- NEMO_DISABLE_CUDA_GRAPHS pattern: MEDIUM — env var confirmed in code and NeMo issues; Blackwell-specific behavior not officially documented
- VRAM budget analysis: LOW-MEDIUM — individual estimates from model cards and parameter counts; simultaneous footprint unmeasured
- Parakeet output format (tuple vs list): MEDIUM — existing code in asr.py handles both, consistent with NeMo 2.0 patterns

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (NeMo and PyTorch move fast — re-verify before lab session if > 30 days)
