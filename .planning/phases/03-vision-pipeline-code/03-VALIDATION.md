---
phase: 3
slug: vision-pipeline-code
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.0.0+ with pytest-asyncio 0.21.0+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) — `asyncio_mode = "auto"` |
| **Quick run command** | `venv/bin/pytest tests/unit/ -x -q` |
| **Full suite command** | `venv/bin/pytest tests/ --cov=smait --cov-report=term-missing` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `venv/bin/pytest tests/unit/ -x -q`
- **After every plan wave:** Run `venv/bin/pytest tests/ --cov=smait --cov-report=term-missing`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | VIS-02 | unit | `venv/bin/pytest tests/unit/test_lip_extractor.py -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 0 | VIS-03 | unit | `venv/bin/pytest tests/unit/test_engagement.py -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 0 | VIS-04 | unit | `venv/bin/pytest tests/unit/test_face_tracker.py -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | VIS-02 | unit | `venv/bin/pytest tests/unit/test_lip_extractor.py::test_lip_roi_output_shape -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | VIS-01 | unit | `venv/bin/pytest tests/unit/test_gaze.py::test_l2cs_step_result_parsed -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | VIS-03 | unit | `venv/bin/pytest tests/unit/test_engagement.py::test_sustained_gaze_reaches_engaged -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | VIS-04 | unit | `venv/bin/pytest tests/unit/test_face_tracker.py::test_iou_high_overlap -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_lip_extractor.py` — stubs for VIS-02 (shape, buffer, time filter, FACE_LOST cleanup)
- [ ] `tests/unit/test_engagement.py` — stubs for VIS-03 (state transitions, event emission, DOA, walking-past filter)
- [ ] `tests/unit/test_face_tracker.py` — stubs for VIS-04 (IOU math, FACE_LOST events, head pose estimation)

*Existing `tests/unit/test_gaze.py` covers VIS-01 basic tests — add `test_l2cs_step_result_parsed` in Wave 1.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
