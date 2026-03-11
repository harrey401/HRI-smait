---
phase: 7
slug: gpu-validation-model-loading
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (unit tests) + standalone GPU smoke scripts |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ --cov=smait --cov-report=term-missing` |
| **GPU validation command** | `python scripts/validate_gpu.py` |
| **Estimated runtime** | ~30s (unit tests) / ~120s (GPU smoke tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ --cov=smait --cov-report=term-missing`
- **Phase gate (GPU):** All smoke scripts pass on RTX 5070 before Phase 7 complete
- **Before `/gsd:verify-work`:** Full suite + GPU smoke scripts must be green
- **Max feedback latency:** 30 seconds (unit tests) / 120 seconds (GPU smoke)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | ENV-01 | smoke (GPU) | `python scripts/smoke_torch.py` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | ENV-01 | smoke (GPU) | `python scripts/smoke_torch.py` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | ASR-01 | smoke (GPU) | `python scripts/smoke_parakeet.py` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | ENV-02 | smoke (GPU) | `python scripts/smoke_all_models.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/smoke_torch.py` — ENV-01: PyTorch sm_120 verification
- [ ] `scripts/smoke_parakeet.py` — ASR-01: Parakeet TDT GPU transcription
- [ ] `scripts/smoke_all_models.py` — ENV-02: simultaneous VRAM budget check
- [ ] `scripts/test_audio/sample_16k.wav` — 3-5 second real speech sample for ASR
- [ ] `scripts/validate_gpu.py` — master script running all smoke tests

*Existing pytest infrastructure covers unit test regressions.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| nvidia-smi shows <12GB total | ENV-02 | Requires physical GPU | Run `nvidia-smi` after `scripts/smoke_all_models.py`, verify "Memory-Usage" column |
| Model outputs are semantically correct | ENV-02 | Human judgement on ASR text | Read Parakeet transcript of known audio, verify it matches expected text |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
