---
phase: 4
slug: speaker-separation-code
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.2 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py tests/unit/test_engagement.py tests/unit/test_audio_pipeline.py -x -q` |
| **Full suite command** | `./venv/bin/python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py tests/unit/test_engagement.py tests/unit/test_audio_pipeline.py -x -q`
- **After every plan wave:** Run `./venv/bin/python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 0 | SEP-04 | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_separate_without_lip_frames_uses_passthrough -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 0 | SEP-06 | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_dolphin_exception_falls_back_to_passthrough -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 0 | SEP-05 | unit | `./venv/bin/python -m pytest tests/unit/test_engagement.py::test_doa_angle_disambiguates_multiple_faces -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 0 | AUD-05 | unit | `./venv/bin/python -m pytest tests/unit/test_audio_pipeline.py::test_vad_emits_segment_after_silence -x` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 0 | AUD-05 | unit | `./venv/bin/python -m pytest tests/unit/test_audio_pipeline.py::test_ring_buffer_extract_aligned -x` | ❌ W0 | ⬜ pending |
| 04-01-06 | 01 | 0 | AUD-05 | unit | `./venv/bin/python -m pytest tests/unit/test_audio_pipeline.py::test_short_segment_rejected -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | SEP-01 | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_look2hear_importable -x` | ✅ | ⬜ pending |
| 04-02-02 | 02 | 1 | SEP-02 | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_run_dolphin_audio_shape -x` | ✅ | ⬜ pending |
| 04-02-03 | 02 | 1 | SEP-03 | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_run_dolphin_video_shape -x` | ✅ | ⬜ pending |
| 04-02-04 | 02 | 1 | SEP-06 | unit | `./venv/bin/python -m pytest tests/unit/test_dolphin_separator.py::test_passthrough_returns_mono -x` | ✅ | ⬜ pending |
| 04-03-01 | 03 | 1 | SEP-04 | unit | `./venv/bin/python -m pytest tests/unit/test_lip_extractor.py -x` | partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_audio_pipeline.py` — stubs for AUD-05 (VAD segmentation, ring buffer, silence threshold, short segment rejection)
- [ ] `tests/unit/test_dolphin_separator.py::test_separate_without_lip_frames_uses_passthrough` — SEP-04/SEP-06 interaction
- [ ] `tests/unit/test_dolphin_separator.py::test_dolphin_exception_falls_back_to_passthrough` — SEP-06 error path
- [ ] `tests/unit/test_engagement.py::test_doa_angle_disambiguates_multiple_faces` — SEP-05

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dolphin GPU inference produces audible separated speech | SEP-01/02/03 | Requires pretrained weights + GPU + real audio | Play test WAV through pipeline, listen to output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
