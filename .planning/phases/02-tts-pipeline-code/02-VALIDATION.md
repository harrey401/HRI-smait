---
phase: 2
slug: tts-pipeline-code
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| 02-01-03 | 01 | 1 | TTS-01 | unit (mock) | `pytest tests/unit/test_tts.py::test_pcm_conversion_correct -x` | NO — W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | TTS-01 | unit (mock) | `pytest tests/unit/test_tts.py::test_torch_tensor_handled -x` | NO — W0 | ⬜ pending |
| 02-02-01 | 02 | 1 | TTS-02 | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_emits_interleaved -x` | NO — W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | TTS-02 | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_flushes_remainder -x` | NO — W0 | ⬜ pending |
| 02-02-03 | 02 | 1 | TTS-02 | unit (mock) | `pytest tests/unit/test_tts.py::test_streaming_event_order -x` | NO — W0 | ⬜ pending |
| 02-02-04 | 02 | 1 | TTS-02 | unit | `pytest tests/unit/test_tts.py::test_sentence_splitting -x` | NO — W0 | ⬜ pending |
| 02-03-01 | 02 | 1 | TTS-03 | unit | `pytest tests/unit/test_tts.py::test_audio_chunk_is_bytes -x` | NO — W0 | ⬜ pending |
| 02-03-02 | 02 | 1 | TTS-03 | unit | `pytest tests/unit/test_protocol.py::test_tts_audio_frame_encoding -x` | NO — W0 | ⬜ pending |
| 02-03-03 | 02 | 1 | TTS-03 | unit (mock ws) | `pytest tests/unit/test_connection_manager.py::test_tts_audio_forwarded -x` | NO — W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_tts.py` — add: `test_pcm_conversion_correct`, `test_torch_tensor_handled`, `test_streaming_emits_interleaved`, `test_streaming_flushes_remainder`, `test_streaming_event_order`, `test_sentence_splitting`, `test_audio_chunk_is_bytes`
- [ ] `tests/unit/test_protocol.py` — NEW file: `test_tts_audio_frame_encoding` (TTS-03 protocol verification)
- [ ] `tests/unit/test_connection_manager.py` — NEW file: `test_tts_audio_forwarded` (TTS-03 ConnectionManager integration)

*Existing infrastructure covers pytest framework; new test files needed for TTS-03.*

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
