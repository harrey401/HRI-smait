---
phase: 2
slug: tts-pipeline-code
status: draft
nyquist_compliant: true
wave_0_complete: true
wave_0_note: "Tests are written inside TDD tasks (RED phase first, then GREEN). Each task's tdd=true attribute enforces write-tests-first within the task. No separate Wave 0 plan needed."
created: 2026-03-09
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.0.0+ with pytest-asyncio 0.21.0+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) — `asyncio_mode = "auto"` |
| **Quick run command** | `pytest tests/unit/test_tts.py -x -q` |
| **Full suite command** | `pytest tests/ --cov=smait --cov-report=term-missing` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit/test_tts.py -x -q`
- **After every plan wave:** Run `pytest tests/ --cov=smait --cov-report=term-missing`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | TTS-01 | unit (mock) | `pytest tests/unit/test_tts.py::test_correct_class_imported -x` | YES | ⬜ pending |
| 02-01-02 | 01 | 1 | TTS-01 | unit (mock) | `pytest tests/unit/test_tts.py::test_synthesize_uses_generator -x` | YES | ⬜ pending |
| 02-01-03 | 01 | 1 | TTS-01 | unit (mock) | `pytest tests/unit/test_tts.py::test_pcm_conversion_correct -x` | NO — TDD in-task | ⬜ pending |
| 02-01-04 | 01 | 1 | TTS-01 | unit (mock) | `pytest tests/unit/test_tts.py::test_torch_tensor_handled -x` | NO — TDD in-task | ⬜ pending |
| 02-01-05 | 01 | 1 | TTS-02 | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_emits_interleaved -x` | NO — TDD in-task | ⬜ pending |
| 02-01-06 | 01 | 1 | TTS-02 | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_flushes_remainder -x` | NO — TDD in-task | ⬜ pending |
| 02-01-07 | 01 | 1 | TTS-02 | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_event_order -x` | NO — TDD in-task | ⬜ pending |
| 02-01-08 | 01 | 1 | TTS-02 | unit | `pytest tests/unit/test_tts.py::test_sentence_splitting -x` | NO — TDD in-task | ⬜ pending |
| 02-01-09 | 01 | 1 | TTS-03 | unit | `pytest tests/unit/test_tts.py::test_audio_chunk_is_bytes -x` | NO — TDD in-task | ⬜ pending |
| 02-02-01 | 02 | 1 | TTS-03 | unit | `pytest tests/unit/test_protocol.py::test_tts_audio_frame_encoding -x` | NO — TDD in-task | ⬜ pending |
| 02-02-02 | 02 | 1 | TTS-03 | unit (mock ws) | `pytest tests/unit/test_connection_manager.py::test_tts_audio_forwarded -x` | NO — TDD in-task | ⬜ pending |
| 02-02-03 | 02 | 1 | TTS-03 | unit (mock ws) | `pytest tests/unit/test_connection_manager.py::test_tts_audio_no_client -x` | NO — TDD in-task | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

All tests are created via TDD-within-task (each task has `tdd="true"` and a `<behavior>` block). The RED phase of each task creates the test stubs first, then the GREEN phase implements production code. No separate Wave 0 plan is needed.

Test files created during execution:
- `tests/unit/test_tts.py` — extended by plan 02-01 task 1 (8 new tests)
- `tests/unit/test_protocol.py` — created by plan 02-02 task 1 (3 new tests)
- `tests/unit/test_connection_manager.py` — created by plan 02-02 task 1 (3 new tests)

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or TDD in-task pattern
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covered by TDD in-task pattern (tests written in RED phase before implementation)
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
