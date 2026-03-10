# Technology Stack: HRI Audio-Visual ML Pipeline

**Project:** SMAIT v3 - Hardware Integration
**Researched:** 2026-03-09
**Overall confidence:** MEDIUM (Dolphin installation is non-trivial; PyTorch Blackwell support still maturing)

## Recommended Stack

### Core ML Models

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Dolphin AV-TSE | HEAD (no releases) | Audio-visual speaker extraction | SOTA AV separation, 6x faster than IIANet, 50% fewer params. Only viable open AV-TSE model with pretrained weights. | MEDIUM |
| Kokoro-82M | 0.9.4 (pip) | Text-to-speech | 82M params, ~1GB VRAM, 24kHz output, Apache-2.0. `KPipeline` generator API enables sentence-level streaming. | HIGH |
| L2CS-Net | HEAD (GitHub) | Gaze estimation | Lightweight ResNet50 backbone, ~0.3GB VRAM. Pipeline API matches existing stub code. | MEDIUM |
| Silero VAD | 6.2.1 (pip) | Voice activity detection | Enterprise-grade, <1ms per chunk on CPU, trained on 6000+ languages. Already integrated in codebase. | HIGH |
| Parakeet TDT | 0.6b-v2/v3 | ASR (speech-to-text) | NVIDIA NeMo, ~2GB VRAM, up to 24min segments. Best open ASR for English. | MEDIUM |

### Audio-Visual Separation: Dolphin Deep Dive

**CRITICAL FINDING: Dolphin is NOT pip-installable.** The stub code's `from dolphin import DolphinModel` and `DolphinModel.from_pretrained("JusperLee/Dolphin")` are WRONG. Here is the actual situation:

#### Actual Repository Structure
- **No `setup.py` or `pyproject.toml`** -- cannot `pip install git+https://github.com/JusperLee/Dolphin.git`
- Contains a local `look2hear/` directory (bundled package, NOT the separate Look2hear repo which says "coming soon")
- Inference entry point: `Inference.py` (designed for video files, not real-time streaming)

#### Actual Model Loading API
```python
# CORRECT import (from cloned repo's local look2hear package)
from look2hear.models import Dolphin

# Load pretrained weights from HuggingFace Hub
audiomodel = Dolphin.from_pretrained("JusperLee/Dolphin")
audiomodel = audiomodel.to(device)
audiomodel.eval()
```

#### Actual Input/Output Tensor Shapes
```python
# Audio: [batch, samples] at 16kHz (NOT [batch, channels, samples])
audio_tensor = mix[None]  # shape: [1, num_samples]

# Video (mouth ROI): [batch, 1, frames, H, W, C] grayscale lip crops
# Mouth ROIs are 88x88 grayscale, preprocessed via look2hear pipelines
video_tensor = torch.from_numpy(
    mouth_roi[None, None]  # shape: [1, 1, num_frames, 88, 88, 1]
).float().to(device)

# Forward pass
est_sources = audiomodel(audio_tensor, video_tensor)
# Output: [1, num_samples] separated audio at 16kHz
```

#### Actual Inference Pipeline
1. Input video is standardized to 25 FPS
2. Audio extracted at 16kHz
3. Face detection via RetinaFace + face_alignment
4. Mouth ROI extraction via landmark-based affine crop (88x88 grayscale)
5. Sliding window: 4-second audio windows with Hann overlap-add
6. Model processes each window: `audiomodel(audio_window, lip_window)`
7. Output is overlap-added back to full-length separated audio

#### Installation Method (Verified)
```bash
git clone https://github.com/JusperLee/Dolphin.git
cd Dolphin
pip install -r requirements.txt
# Key deps: torch, torchaudio, torchvision, moviepy, face_alignment,
#   beartype, taylor_series_linear_attention, huggingface_hub, einops,
#   vector_quantize_pytorch, retina-face, safetensors, tf-keras
```

Then either:
- Add the Dolphin repo root to `PYTHONPATH`, OR
- Copy the `look2hear/` directory into the SMAIT project, OR
- Create a `setup.py` wrapper to make it importable

#### What the Stub Code Gets Wrong

| Stub Code | Reality | Fix |
|-----------|---------|-----|
| `from dolphin import DolphinModel` | `from look2hear.models import Dolphin` | Change import |
| `DolphinModel.from_pretrained(...)` | `Dolphin.from_pretrained(...)` | Change class name |
| Audio shape: `(batch, channels, samples)` | Audio shape: `(batch, samples)` mono only | Remove channel dim |
| Video shape: `(batch, frames, H, W, C)` | Video shape: `(batch, 1, frames, 88, 88, 1)` | Add extra dim, fix size |
| Expects multi-channel input | Model is mono-only (single channel audio) | Beamform/mix-down BEFORE Dolphin |
| `model(audio)` audio-only mode | Not supported -- always needs video tensor | Always provide lip ROI |

#### VRAM Estimate
- Model weights: ~1-2GB (82M-class architecture)
- Inference buffer: ~0.5GB for 4-second windows
- **Total: ~2-3GB** (within 12GB budget)

### Kokoro-82M TTS

**Status: pip-installable, API is different from stub code.**

#### Installation
```bash
pip install kokoro>=0.9.4 soundfile
# For phonemizer: needs espeak-ng system package
# sudo apt install espeak-ng
```

#### Actual API (stub code is WRONG)
```python
# WRONG (stub): from kokoro import KokoroTTS; model = KokoroTTS()
# CORRECT:
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')  # 'a' = American English

# Generate audio -- returns a GENERATOR (sentence-level streaming built-in)
for graphemes, phonemes, audio in pipeline(text, voice='af_heart', speed=1.0):
    # audio is a numpy float32 array at 24kHz
    # Convert to int16 PCM: (audio * 32767).clip(-32768, 32767).astype(np.int16)
    pass
```

#### Key Differences from Stub

| Stub Code | Reality | Fix |
|-----------|---------|-----|
| `from kokoro import KokoroTTS` | `from kokoro import KPipeline` | Change import |
| `KokoroTTS()` constructor | `KPipeline(lang_code='a')` | Change init |
| `model.generate(text)` returns audio | `pipeline(text, voice='af_heart')` returns generator | Use generator pattern |
| No voice selection | Voice param required: `af_heart`, `am_adam`, etc. | Add voice config |

#### Available Voices
- **American Female:** af_heart, af_bella, af_nicole, af_sarah, af_sky
- **American Male:** am_adam, am_michael
- **British Female:** bf_emma, bf_isabella
- **British Male:** bm_george, bm_lewis

#### Streaming Strategy
KPipeline already yields audio per sentence/phrase. The existing `TTSEngine.speak_streaming()` architecture (buffer tokens, detect sentence boundary, synthesize) maps well -- just replace the inner `synthesize()` call with `pipeline()`.

**Sample rate:** 24kHz (matches config).
**VRAM:** ~1GB.

### L2CS-Net Gaze Estimation

**Status: GitHub-installable from fork, not on PyPI.**

#### Installation
```bash
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```
Note: This is a maintained fork by edavalosanaya, not the original Ahmednull repo (which may lack pip packaging).

#### API (stub code is approximately correct)
```python
from l2cs import Pipeline as L2CSPipeline

gaze_pipeline = L2CSPipeline(
    weights=None,          # Auto-download from Google Drive
    arch='ResNet50',       # or 'Gaze360'
    device=torch.device('cuda')
)

# Inference: pass a face crop (BGR numpy array)
results = gaze_pipeline.step(face_crop_bgr)
yaw = float(results.yaw[0])    # degrees
pitch = float(results.pitch[0])  # degrees
```

#### Stub Accuracy
The existing `gaze.py` stub is close to correct. Minor fixes needed:
- `arch` param in stub says `"Gaze360"` -- verify if this refers to the weight set name or architecture. The pipeline accepts `"ResNet50"` as arch with Gaze360-trained weights.
- Weights auto-download may fail in air-gapped environments -- pre-download `L2CSNet_gaze360.pkl`.

**VRAM:** ~0.3GB.

### Silero VAD + Turn-Taking

**Status: Production-ready, already integrated.**

#### Installation
```bash
pip install silero-vad>=6.2.1
```

#### Recommended Turn-Taking Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `activation_threshold` | 0.5 | Default, good for most environments |
| `min_silence_duration` | 0.55s | Default balance of responsiveness vs false triggers |
| `min_speech_duration` | 0.05s | Ignore very brief sounds |
| `min_endpointing_delay` | 500ms | Industry standard for conversational agents |
| Hard cutoff silence | 1500ms | Force END_OF_TURN regardless of context |

#### VAD-Based EOU Strategy (replacing LiveKit turn-detector)

The existing `eou_detector.py` should be refactored to a pure VAD-silence approach:

1. **300-500ms silence after speech** -> check heuristic EOU (punctuation-based)
2. **If P(eou) > 0.7** -> emit END_OF_TURN
3. **700ms silence** -> emit END_OF_TURN regardless (conservative)
4. **1500ms silence** -> hard cutoff (safety net)

This is simpler and more reliable than the LiveKit transformer model, which is unavailable anyway. The heuristic EOU in the existing code is actually reasonable for a first pass.

**For noisy conference rooms:** Increase `activation_threshold` to 0.6 and `min_silence_duration` to 0.75s.

### Parakeet TDT ASR on Blackwell

**Status: Works with workarounds. MEDIUM confidence.**

#### Known Issues with sm_120
- PyTorch stable (as of March 2026) still does NOT include sm_120 kernels
- PyTorch nightly with CUDA 12.8+ or 12.9 can detect and run on sm_120
- CUDA graphs MUST be disabled: `NEMO_DISABLE_CUDA_GRAPHS=1`

#### Recommended Setup
```bash
# Install PyTorch nightly with CUDA 12.8
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install NeMo
pip install nemo_toolkit[asr]

# Runtime env var (CRITICAL)
export NEMO_DISABLE_CUDA_GRAPHS=1
```

#### Risk
- Nightly PyTorch may have regressions
- NeMo's internal torch.compile / Triton calls may fail with "sm_120 is not defined for option 'gpu-name'" errors
- **Mitigation:** Pin a known-working nightly date; test thoroughly before deployment

**VRAM:** ~2GB.

## Audio-Visual Separation Alternatives (If Dolphin Fails)

If Dolphin integration proves too difficult (non-pip-installable, video-file-oriented API, no streaming), here are ranked alternatives:

| Alternative | Approach | Real-time? | Quality | Availability | Confidence |
|-------------|----------|------------|---------|--------------|------------|
| **Beamforming-only (CAE)** | Skip AV separation, use CAE beamformed audio directly | Yes | Moderate | Already have it | HIGH |
| **AV-SepFormer** | Cross-attention Transformer for AV separation | With effort | Good | Research code only (ICASSP 2023) | LOW |
| **IIANet** | Iterative AV separation | No (too slow, 6x slower than Dolphin) | Best | Research code | LOW |
| **VisualVoice** | Audio-visual speech separation (UT Austin/Meta) | With effort | Good | GitHub available | LOW |
| **Audio-only SepFormer** | SpeechBrain SepFormer, no visual cues | Yes | Moderate | pip install speechbrain | MEDIUM |

**Recommendation:** If Dolphin integration takes >2 days, fall back to CAE beamforming + audio-only enhancement. The CAE already provides directional audio with noise reduction. Dolphin adds speaker isolation via lip correlation, but the CAE + DOA angles may be sufficient for single-target-speaker scenarios.

## Alternatives NOT to Use

| Technology | Why Not |
|------------|---------|
| LiveKit turn-detector | Private/unavailable package, HuggingFace model also inaccessible |
| Piper TTS | Replaced by Kokoro (better quality, similar VRAM) |
| Google Looking to Listen | No public weights, 2018 architecture, not reproducible |
| Whisper for ASR | Parakeet TDT is faster and more accurate for English; Whisper wastes VRAM |
| WebRTC VAD | Silero VAD is strictly better (trained on 6000+ languages, higher accuracy) |

## VRAM Budget

| Model | Estimated VRAM | Notes |
|-------|---------------|-------|
| Parakeet TDT 0.6B | ~2.0 GB | ASR |
| Dolphin AV-TSE | ~2.0-3.0 GB | Speaker separation |
| Kokoro-82M | ~1.0 GB | TTS |
| L2CS-Net | ~0.3 GB | Gaze estimation |
| Phi-4 Mini (Q4) | ~3.0 GB | LLM (via Ollama) |
| MediaPipe Face Mesh | ~0.1 GB | Face tracking (mostly CPU) |
| Silero VAD | ~0.0 GB | CPU only |
| **Total** | **~8.4-9.4 GB** | **Within 12GB budget** |

## Installation Script

```bash
# Core PyTorch (nightly for Blackwell sm_120)
pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# NeMo ASR
pip install nemo_toolkit[asr]

# Kokoro TTS
pip install kokoro>=0.9.4 soundfile
sudo apt install espeak-ng  # for phonemizer

# L2CS-Net gaze
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main

# Silero VAD
pip install silero-vad>=6.2.1

# Dolphin (NOT pip installable -- manual setup)
git clone https://github.com/JusperLee/Dolphin.git /opt/dolphin
cd /opt/dolphin && pip install -r requirements.txt
# Then either:
#   export PYTHONPATH=/opt/dolphin:$PYTHONPATH
#   OR symlink look2hear into the project

# Supporting
pip install opencv-python>=4.8.0 numpy>=1.24.0 mediapipe>=0.10.0
pip install openai>=1.0.0 websockets>=12.0 uvloop>=0.19.0
pip install python-dotenv psutil einops

# Runtime environment
export NEMO_DISABLE_CUDA_GRAPHS=1
```

## Sources

### Verified (HIGH confidence)
- [Kokoro PyPI](https://pypi.org/project/kokoro/) -- v0.9.4, Python 3.10-3.12
- [Silero VAD PyPI](https://pypi.org/project/silero-vad/) -- v6.2.1
- [Kokoro-82M HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) -- API docs, voice list
- [hexgrad/kokoro GitHub](https://github.com/hexgrad/kokoro) -- KPipeline API

### Verified (MEDIUM confidence)
- [JusperLee/Dolphin GitHub](https://github.com/JusperLee/Dolphin) -- repo structure, requirements.txt
- [JusperLee/Dolphin HuggingFace](https://huggingface.co/JusperLee/Dolphin) -- model weights (conf.yml + best_model.pth)
- [Dolphin HF Space](https://huggingface.co/spaces/JusperLee/Dolphin) -- Inference.py with actual API
- [L2CS-Net GitHub](https://github.com/Ahmednull/L2CS-Net) -- Pipeline API, weight download
- [edavalosanaya/L2CS-Net fork](https://github.com/edavalosanaya/L2CS-Net) -- pip-installable fork
- [PyTorch sm_120 issue](https://github.com/pytorch/pytorch/issues/164342) -- Blackwell support status

### Community sources (LOW confidence)
- [LiveKit VAD docs](https://docs.livekit.io/agents/logic/turns/vad/) -- silence threshold recommendations
- [LiveKit turn detection blog](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection/) -- EOU approach patterns
- [PyTorch Blackwell workaround](https://medium.com/@harishpillai1994/fix-pytorch-sm-120-on-rtx-blackwell-gpus-cuda-docker-cu128-setup-to-run-llms-44f25179ac76)

---

*Stack research: 2026-03-09*
