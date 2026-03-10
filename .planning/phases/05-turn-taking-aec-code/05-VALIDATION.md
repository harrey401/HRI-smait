---
phase: 5
slug: turn-taking-aec-code
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` asyncio_mode = "auto" |
| **Quick run command** | `./venv/bin/python -m pytest tests/unit/test_eou_detector.py tests/unit/test_transcriber.py tests/unit/test_barge_in.py tests/unit/test_aec.py -x -q` |
| **Full suite command** | `./venv/bin/python -m pytest tests/ -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./venv/bin/python -m pytest tests/unit/test_eou_detector.py tests/unit/test_transcriber.py tests/unit/test_barge_in.py tests/unit/test_aec.py -x -q`
- **After every plan wave:** Run `./venv/bin/python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 0 | ASR-02 | unit | `pytest tests/unit/test_transcriber.py -x` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 0 | AUD-07 | unit | `pytest tests/unit/test_barge_in.py -x` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 0 | AUD-06 | unit | `pytest tests/unit/test_aec.py -x` | ❌ W0 | ⬜ pending |
| 05-01-04 | 01 | 1 | ASR-03 | unit | `pytest tests/unit/test_eou_detector.py -x` | ✅ extends | ⬜ pending |
| 05-01-05 | 01 | 1 | ASR-03 | unit | `pytest tests/unit/test_eou_detector.py -x` | ✅ extends | ⬜ pending |
| 05-02-01 | 02 | 1 | ASR-02 | unit | `pytest tests/unit/test_transcriber.py -x` | ❌ W0 | ⬜ pending |
| 05-02-02 | 02 | 1 | ASR-02 | unit | `pytest tests/unit/test_transcriber.py -x` | ❌ W0 | ⬜ pending |
| 05-02-03 | 02 | 2 | AUD-06 | unit | `pytest tests/unit/test_aec.py -x` | ❌ W0 | ⬜ pending |
| 05-02-04 | 02 | 2 | AUD-07 | unit | `pytest tests/unit/test_barge_in.py -x` | ❌ W0 | ⬜ pending |
| 05-02-05 | 02 | 2 | AUD-07 | unit | `pytest tests/unit/test_barge_in.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_transcriber.py` — stubs for ASR-02 (hallucination filter, NeMo hypothesis path, confidence thresholds)
- [ ] `tests/unit/test_barge_in.py` — stubs for AUD-07 (barge-in state transitions, TTS cancellation, TTS_END guarantee)
- [ ] `tests/unit/test_aec.py` — stubs for AUD-06 (SoftwareAEC frame processing, CAE status gating)
- [ ] `EventType.BARGE_IN` — add to `smait/core/events.py`
- [ ] `speexdsp` install: `pip install speexdsp==0.1.1` (libspeexdsp-dev already installed)

*Existing `tests/unit/test_eou_detector.py` covers ASR-03 base — extends with VAD rewrite cases.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| AEC quality with CAE hardware | AUD-06 | Requires physical CAE device | Phase 7 (LAB) — play TTS through speaker, verify echo removal |
| Barge-in latency perception | AUD-07 | Subjective UX timing | Speak during TTS, verify <500ms response |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
