# Domain Pitfalls

**Domain:** HRI audio-visual hardware integration (SMAIT v3)
**Researched:** 2026-03-09 (updated with verified API findings)

## Critical Pitfalls

Mistakes that cause rewrites, multi-day delays, or system-level failures.

### Pitfall 1: Blackwell sm_120 GPU Architecture Breaks PyTorch Ecosystem

**What goes wrong:** The RTX 5070 uses NVIDIA Blackwell sm_120 compute architecture. Stable PyTorch releases through March 2026 do NOT ship with sm_120 kernels. Installing PyTorch via `pip install torch` pulls CUDA 12.6 or earlier builds that produce: `CUDA error: no kernel image is available for execution on the device`.

**Why it happens:** PyTorch binary distributions are compiled against specific CUDA compute capabilities (sm_50 through sm_90). Blackwell sm_120 requires CUDA 12.8+ builds. Third-party libraries (NeMo, transformers, torchaudio) may pin or downgrade PyTorch to incompatible versions during installation.

**Consequences:** Complete GPU failure. All five ML models refuse to run. System falls back to CPU or crashes.

**Prevention:**
- Pin PyTorch nightly with explicit `--index-url https://download.pytorch.org/whl/nightly/cu128` (CUDA 12.8+).
- After every `pip install` of any ML library, verify: `python -c "import torch; print(torch.cuda.get_arch_list())"` must include `sm_120`.
- Set `NEMO_DISABLE_CUDA_GRAPHS=1` -- Blackwell has known issues with CUDA graphs in NeMo.
- Test NeMo Parakeet specifically: `model.transcribe(["test.wav"])` as NeMo has its own CUDA kernel compilation paths.

**Detection:** Run `torch.cuda.get_arch_list()` at startup. If `sm_120` is absent, abort with clear error.

**Confidence:** HIGH -- multiple PyTorch forum threads, GitHub issues confirm active problem.

**Sources:**
- [PyTorch Forums: RTX 5070 Ti sm_120](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509)
- [PyTorch Issue #164342: sm_120 support](https://github.com/pytorch/pytorch/issues/164342)
- [Fix PyTorch sm_120 on Blackwell GPUs](https://medium.com/@harishpillai1994/fix-pytorch-sm-120-on-rtx-blackwell-gpus-cuda-docker-cu128-setup-to-run-llms-44f25179ac76)

---

### Pitfall 2: Dolphin Stub Code is Entirely Wrong (VERIFIED)

**What goes wrong:** The current `dolphin_separator.py` uses a fabricated API that does not exist in the actual Dolphin repository. Every line of the model interaction code is wrong.

**Why it happens:** Stub was written speculatively without access to the actual repo. The import path, class name, tensor shapes, and even the multi-channel capability are all invented.

**Consequences:** ImportError on first run (`from dolphin import DolphinModel` -- no such module). Even if imports were somehow fixed, tensor shape mismatches would cause crashes or silent wrong results.

**Specific errors verified against HF Space Inference.py:**

| Stub Code | Actual API | Impact |
|-----------|-----------|--------|
| `from dolphin import DolphinModel` | `from look2hear.models import Dolphin` | ImportError |
| `DolphinModel.from_pretrained("JusperLee/Dolphin")` | `Dolphin.from_pretrained("JusperLee/Dolphin")` | AttributeError |
| Audio: `(batch, channels, samples)` | Audio: `(batch, samples)` mono only | Shape mismatch |
| Video: `(batch, frames, H, W, C)` | Video: `(batch, 1, frames, 88, 88, 1)` grayscale | Shape mismatch |
| `model(audio)` audio-only mode | Not supported -- always needs video tensor | Runtime error |
| `pip install git+...` | Not pip-installable (no setup.py/pyproject.toml) | Install failure |

**Prevention:** Rewrite `dolphin_separator.py` from scratch. Installation: `git clone https://github.com/JusperLee/Dolphin.git` then `pip install -r requirements.txt` then add repo to PYTHONPATH or symlink `look2hear/` into project.

**Detection:** Immediate ImportError on `from dolphin import DolphinModel`.

**Confidence:** HIGH -- verified directly from HuggingFace Space source code.

---

### Pitfall 3: Dolphin Takes Mono Audio Only (Multi-Channel Assumption Wrong)

**What goes wrong:** The stub assumes Dolphin accepts 4-channel audio from the mic array. It does not. Dolphin takes mono audio at 16kHz only.

**Why it happens:** Confusion between "audio-visual" (uses lips + audio) and "multi-channel" (uses mic array spatially). These are orthogonal capabilities.

**Consequences:** The 4-channel raw audio must be beamformed or mixed down to mono BEFORE Dolphin. The CAE output IS the mono input for Dolphin. The `used_multichannel` flag in `SeparationResult` is meaningless.

**Prevention:** Audio pipeline order: Raw 4-ch -> CAE beamform -> mono 16kHz -> Dolphin (with lip ROI).

**Detection:** Shape mismatch error if multi-channel tensor is passed to Dolphin.

**Confidence:** HIGH -- verified from Inference.py: `mix = mix.mean(dim=0)` (mono mixdown before model).

---

### Pitfall 4: Kokoro API Mismatch (VERIFIED)

**What goes wrong:** Stub uses `from kokoro import KokoroTTS` and `model.generate(text)`. Neither exists. The real API is completely different.

**Actual API:**
```python
from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')  # 'a' = American English
for graphemes, phonemes, audio in pipeline(text, voice='af_heart', speed=1.0):
    # audio is numpy float32 at 24kHz
```

**Consequences:** TTS silently falls back to Android TTS for every utterance. System "works" but with degraded quality and no streaming. The sentence-level streaming architecture is sound but the implementation code needs rewriting.

**Prevention:** Use verified `KPipeline` generator API. The generator pattern naturally maps to the existing sentence-streaming architecture.

**Confidence:** HIGH -- verified on PyPI (v0.9.4) and HuggingFace docs.

---

### Pitfall 5: VRAM Exhaustion and Memory Fragmentation

**What goes wrong:** Loading 5+ models into 12GB GPU causes fragmentation. PyTorch's caching allocator splits freed memory into small unusable segments. OOM errors occur when 3-4GB appears free but no contiguous block is large enough.

**Prevention:**
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Load models largest-first: Phi-4 -> Parakeet -> Dolphin -> Kokoro -> L2CS -> Silero
- Use `torch.inference_mode()` (not just `torch.no_grad()`) for all inference
- Log VRAM after each model load

**Detection:** Run `torch.cuda.max_memory_allocated()` after loading all models + one inference pass each. If > 10.5GB, budget is too tight.

**Confidence:** HIGH -- well-documented PyTorch behavior.

---

### Pitfall 6: Audio-Visual Sync Drift Between Streams

**What goes wrong:** Audio (WebSocket 0x01/0x03) and video (0x02) travel different paths with different latencies. Over minutes, clock drift causes 50-200ms desync. Dolphin requires aligned lip video and speech audio -- even 100ms drift degrades separation.

**Prevention:**
- Timestamp frames at Android source using `SystemClock.elapsedRealtimeNanos()` (monotonic)
- On server, use timestamp-based ring buffers for alignment
- Implement drift detector: warn at 80ms, resync at 200ms
- Do NOT rely on server arrival time as proxy for capture time

**Confidence:** MEDIUM -- standard AV sync issue, magnitude depends on WiFi/device.

---

## Moderate Pitfalls

### Pitfall 7: Dolphin 4-Second Window Adds Real-Time Latency

**What goes wrong:** Dolphin processes 4-second audio windows with overlap-add. Waiting for 4 seconds of audio before processing is unacceptable for real-time conversation.

**Prevention:** Investigate shorter windows (1-2 seconds). The overlap-add with Hann windowing should work at shorter durations with some quality degradation. Alternatively, use a sliding buffer that processes on each new chunk arrival.

### Pitfall 8: Mouth ROI Mismatch with Dolphin Expectations

**What goes wrong:** Existing `LipExtractor` produces 96x96 RGB crops. Dolphin expects 88x88 grayscale preprocessed via `look2hear`'s validation pipeline.

**Prevention:** Study `get_preprocessing_pipelines()["val"]` from Dolphin repo. Add preprocessing in DolphinSeparator (resize to 88x88, convert grayscale, normalize) rather than changing LipExtractor.

### Pitfall 9: L2CS-Net Weight Download Fails in Production

**What goes wrong:** Weights hosted on Google Drive with rate limits.

**Prevention:** Pre-download `L2CSNet_gaze360.pkl` during setup. Point `weights=` to local path.

### Pitfall 10: LiveKit EOU Model is Completely Unavailable

**What goes wrong:** Both `livekit.plugins.turn_detector` and `livekit/turn-detector` HuggingFace model are private/unavailable.

**Prevention:** Replace with VAD silence-based approach. The existing heuristic EOU (punctuation-based) is actually reasonable as a first pass. Recommended thresholds: 500ms min silence for EOU check, 700ms for auto-trigger, 1500ms hard cutoff.

### Pitfall 11: CAE Branch Merge Silently Drops Changes

**What goes wrong:** `cae-work-march2` branch was reverted on main. Git merge silently excludes the original changes because the revert is "newer."

**Prevention:** Revert-the-revert: `git revert <revert-commit>` on a new branch, then merge/cherry-pick. Verify beamforming code present post-merge.

### Pitfall 12: Synchronous ML Inference Blocks Async Event Loop

**What goes wrong:** GPU inference calls (50-400ms) in async handlers block the entire event loop. Video frames pile up, WebSocket backpressure builds.

**Prevention:** Wrap inference > 50ms in `asyncio.to_thread()` or `run_in_executor()`. Use dedicated thread pool for GPU operations.

### Pitfall 13: NeMo 2.0 API Instability

**What goes wrong:** NeMo 2.0 changed `transcribe()` return type. Future updates may break again. NeMo's dependencies can conflict with Dolphin/Kokoro.

**Prevention:** Pin NeMo to exact version. Add startup transcription test. Install NeMo before other ML libraries to avoid downgrades.

### Pitfall 14: 25fps Video Sync Assumption

**What goes wrong:** Dolphin expects 25fps lip video. Android camera captures at 30fps or variable rate.

**Prevention:** Downsample to 25fps or pass variable-length sequence (try variable first).

## Minor Pitfalls

### Pitfall 15: espeak-ng Not Installed for Kokoro
Kokoro's phonemizer falls back to basic G2P without espeak-ng. Fix: `sudo apt install espeak-ng`.

### Pitfall 16: Face Alignment Dependency Conflicts
Dolphin requires `face_alignment` + `retina-face` which may conflict with MediaPipe. Consider extracting mouth ROIs from MediaPipe landmarks instead.

### Pitfall 17: EventBus Stale Loop Reference
`EventBus._loop` is cached on first `emit()` and never updated. Breaks test suites with multiple `asyncio.run()` calls. Fix: call `asyncio.get_running_loop()` every time.

### Pitfall 18: Video Pipeline Disconnected from Events
`VideoPipeline.process_jpeg()` is never called from any event handler. Video frames arrive but never reach face tracker. Must wire before any vision model work.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Environment setup | sm_120 incompatibility (#1) | Verify `torch.cuda.get_arch_list()` includes sm_120 |
| Dolphin installation | Not pip-installable (#2) | Clone repo, add look2hear to PYTHONPATH |
| Dolphin integration | Mono-only audio (#3), wrong stub API (#2) | Rewrite from scratch using verified API |
| Kokoro TTS | Wrong API (#4) | Use KPipeline generator pattern |
| Model loading | VRAM exhaustion (#5) | Load largest first, expandable_segments=True |
| AV fusion | Sync drift (#6), 4s window latency (#7) | Timestamp alignment, shorter windows |
| Lip extraction | ROI format mismatch (#8) | Preprocess in DolphinSeparator |
| ASR activation | NeMo API breaks (#13), sm_120 (#1) | Pin NeMo version, startup test |
| Turn-taking | LiveKit unavailable (#10) | VAD silence thresholds |
| Android CAE | Branch merge drops changes (#11), 8ch/4ch (#existing) | Revert-the-revert strategy |
| Event loop | Blocking inference (#12) | asyncio.to_thread() for >50ms ops |
| Test suite | EventBus stale loop (#17) | Fix loop caching first |

## Sources

- [JusperLee/Dolphin GitHub](https://github.com/JusperLee/Dolphin) -- repo structure confirmed
- [Dolphin HF Space](https://huggingface.co/spaces/JusperLee/Dolphin) -- Inference.py with actual API
- [JusperLee/Dolphin HuggingFace](https://huggingface.co/JusperLee/Dolphin) -- model weights (conf.yml + best_model.pth)
- [Kokoro PyPI](https://pypi.org/project/kokoro/) -- v0.9.4, KPipeline API
- [hexgrad/Kokoro-82M HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) -- voice list, API docs
- [PyTorch sm_120 issues](https://github.com/pytorch/pytorch/issues/164342)
- [LiveKit VAD docs](https://docs.livekit.io/agents/logic/turns/vad/) -- silence thresholds
- [Silero VAD PyPI](https://pypi.org/project/silero-vad/) -- v6.2.1

---

*Pitfalls research: 2026-03-09 (updated with verified API findings)*
