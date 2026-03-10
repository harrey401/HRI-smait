---
phase: 1
slug: dependency-setup-stub-api-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.0.0+ with pytest-asyncio 0.21.0+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `pytest tests/unit/ -x -q` |
| **Full suite command** | `pytest tests/ --cov=smait --cov-report=term-missing` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/ -x -q`
- **After every plan wave:** Run `pytest tests/ --cov=smait --cov-report=term-missing`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | QUAL-01 | unit | `pytest tests/unit/ -x -q` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | ENV-03 | unit (import) | `pytest tests/unit/test_dolphin_separator.py::test_look2hear_importable -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | QUAL-01 | unit (mock) | `pytest tests/unit/test_dolphin_separator.py -x` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 1 | QUAL-01 | unit (mock) | `pytest tests/unit/test_tts.py -x` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 1 | QUAL-01 | unit (mock) | `pytest tests/unit/test_gaze.py -x` | ❌ W0 | ⬜ pending |
| 1-03-03 | 03 | 1 | QUAL-01 | unit | `pytest tests/unit/test_eou_detector.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — shared fixtures (config, event_bus, silence_audio, speech_audio)
- [ ] `tests/unit/test_dolphin_separator.py` — covers ENV-03, QUAL-01 (dolphin)
- [ ] `tests/unit/test_tts.py` — covers QUAL-01 (kokoro)
- [ ] `tests/unit/test_gaze.py` — covers QUAL-01 (l2cs)
- [ ] `tests/unit/test_eou_detector.py` — covers QUAL-01 (eou)
- [ ] `pip install pytest-cov` — missing from dev deps; add to `pyproject.toml`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `from look2hear.models import Dolphin` CPU import | ENV-03 | Requires vendored source present | Run `python -c "from look2hear.models import Dolphin; print('OK')"` after vendoring |
| Dolphin dependency conflict check | ENV-03 | Requires real pip install in venv | `pip install -r vendor/dolphin-src/requirements.txt` and check for conflicts |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
