# Research Summary: SMAIT v3 HRI Audio-Visual ML Pipeline

**Domain:** Human-Robot Interaction, Audio-Visual ML Models
**Researched:** 2026-03-09 (updated with verified API findings)
**Overall confidence:** MEDIUM

## Executive Summary

The SMAIT v3 ML pipeline relies on five key models: Dolphin AV-TSE for speaker separation, Kokoro-82M for TTS, L2CS-Net for gaze estimation, Silero VAD for voice activity detection, and Parakeet TDT for ASR. Research reveals that the existing stub code has significant API mismatches with every model except Silero VAD. The most critical finding is that Dolphin is NOT pip-installable and its actual API differs substantially from the stubs -- the import path, class name, tensor shapes, and multi-channel assumption are all wrong.

Dolphin's actual API (verified from HuggingFace Space Inference.py code):
- Import: `from look2hear.models import Dolphin` (not `from dolphin import DolphinModel`)
- Audio input: `[batch, samples]` mono at 16kHz (not multi-channel)
- Video input: `[batch, 1, frames, 88, 88, 1]` grayscale lip ROIs (not `[batch, frames, H, W, C]`)
- Video is REQUIRED -- no audio-only fallback mode
- Installation: `git clone` + `pip install -r requirements.txt` + add to PYTHONPATH (no `pip install git+...`)

Kokoro-82M is the most straightforward integration -- pip-installable (`pip install kokoro>=0.9.4`) with a generator-based API (`KPipeline`) that naturally supports sentence-level streaming. The stub's `KokoroTTS` class does not exist; the real class is `KPipeline(lang_code='a')` yielding `(graphemes, phonemes, audio)` tuples.

L2CS-Net is installable from a maintained fork (`pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main`). The `Pipeline` API in the stub is approximately correct but needs `arch='ResNet50'` not `arch='Gaze360'`.

The Blackwell GPU (RTX 5070, sm_120) remains a compatibility risk. PyTorch stable still lacks sm_120 support as of March 2026, requiring nightly builds with CUDA 12.8+. The workaround is functional but fragile.

VRAM budget of ~8.4-9.4GB across all models fits within the 12GB constraint.

## Key Findings

**Stack:** Dolphin requires manual clone + PYTHONPATH; Kokoro is `pip install kokoro>=0.9.4` with `KPipeline` API; L2CS-Net from GitHub fork; Silero VAD v6.2.1 from PyPI; all stub APIs need fixing.
**Architecture:** Dolphin processes 4-second sliding windows with Hann overlap-add, expects mono 16kHz audio + 88x88 grayscale lip ROIs at 25fps. Beamforming/mix-down must happen BEFORE Dolphin.
**Critical pitfall:** Every stub import and API call is wrong. The stubs will all fail with ImportError or shape mismatch on first real run.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Phase 1: Environment + PyTorch Setup** - Get sm_120 working, verify CUDA
   - Addresses: Blackwell GPU compatibility, VRAM budget validation
   - Avoids: Building on broken GPU foundation

2. **Phase 2: Fix All Stub APIs** - Correct imports, class names, tensor shapes
   - Addresses: Dolphin, Kokoro, L2CS-Net, EOU detector API mismatches
   - Avoids: Building on code that fails at first import

3. **Phase 3: Kokoro TTS Integration** - Lowest risk, highest user-visible impact
   - Addresses: TTS replacement (Piper -> Kokoro), sentence streaming via KPipeline generator
   - Avoids: Dolphin complexity blocking all progress

4. **Phase 4: Vision Pipeline + L2CS-Net** - Enable face tracking and gaze for engagement
   - Addresses: Gaze estimation, lip extraction for Dolphin input
   - Avoids: Dolphin integration without lip frames ready

5. **Phase 5: Dolphin AV-TSE Integration** - Highest complexity, highest value
   - Addresses: Speaker separation, the core innovation
   - Avoids: Premature integration before lip pipeline works

6. **Phase 6: VAD-based Turn-Taking** - Replace LiveKit EOU with silence-based approach
   - Addresses: End-of-utterance detection, conversation flow
   - Avoids: Dependency on unavailable LiveKit package

7. **Phase 7: Full Pipeline Integration + Tuning** - Wire everything, optimize latency
   - Addresses: End-to-end conversation loop, latency optimization to <1500ms
   - Avoids: Integrating untested individual components

**Phase ordering rationale:**
- Environment first because nothing works without sm_120 GPU support
- API fixes before any model work to establish correct interfaces
- Kokoro before Dolphin because it is pip-installable and immediately testable
- Vision pipeline before Dolphin because Dolphin REQUIRES lip frames
- VAD turn-taking is independent and can be done in parallel with Phases 4-5
- Full integration last to avoid debugging multiple broken pieces simultaneously

**Research flags for phases:**
- Phase 5 (Dolphin): NEEDS deeper research -- 4-second window may not work for real-time; mouth ROI preprocessing pipeline needs study
- Phase 1 (PyTorch nightly): Likely needs debugging -- sm_120 + NeMo + Triton is a fragile combination
- Phase 4 (L2CS weights): Weights on Google Drive may be unreliable; pre-download recommended

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Kokoro TTS API | HIGH | Verified on PyPI v0.9.4, KPipeline documented on HuggingFace |
| Silero VAD | HIGH | Mature package v6.2.1, clear API |
| Dolphin API | MEDIUM | Verified from HF Space Inference.py, but real-time streaming integration untested |
| L2CS-Net | MEDIUM | Fork installable, auto-download weights from Google Drive may be flaky |
| Parakeet on Blackwell | MEDIUM | Requires nightly PyTorch, CUDA graphs disabled, active community issues |
| Dolphin installation | MEDIUM | Confirmed no setup.py; `look2hear` is bundled in repo, not a separate package |

## Gaps to Address

- Dolphin real-time streaming: Model designed for video file processing (4-second windows). Real-time adaptation needs prototyping.
- Dolphin `look2hear` preprocessing: The `get_preprocessing_pipelines()["val"]` transform chain is undocumented; need to read source.
- Dolphin VRAM: Paper says 251MB model + runtime. Needs real measurement.
- L2CS-Net weight hosting: Google Drive may rate-limit; consider bundling weights.
- PyTorch nightly pin: Identify and lock a specific nightly date that works across all models + sm_120.
- Mouth ROI from MediaPipe vs RetinaFace: Dolphin's inference uses RetinaFace for face detection; SMAIT uses MediaPipe. Need to verify if MediaPipe landmarks produce compatible mouth ROIs.

---

*Summary: 2026-03-09 (updated with verified API findings)*
