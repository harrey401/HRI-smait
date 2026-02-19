# SMAIT HRI v2.0 — Smart Multimodal AI Interaction for Telepresence

A real-time Human-Robot Interaction framework that enables natural, face-to-face conversations between humans and service robots. Built for the **Jackie** robot platform at **San José State University, BioRob Lab**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![License](https://img.shields.io/badge/License-Research-orange)

## Features

- **Sub-100ms ASR** — NVIDIA Parakeet TDT (fp16 CUDA) or Faster-Whisper fallback
- **Active Speaker Detection** — LASER-based visual speech verification with temporal buffering
- **Face Tracking** — MediaPipe multi-face tracking with head pose estimation
- **Engagement Detection** — Proximity + attention + natural engagement (no wake word needed)
- **Proactive Greeting** — Robot initiates conversation when someone approaches
- **Behavior Trees** — py_trees-based composable robot behaviors
- **Semantic VAD** — Early turn prediction for lower response latency
- **Multi-turn Dialogue** — Sliding window conversation memory with LLM backends (OpenAI, Ollama)
- **Local TTS** — Piper TTS (~70ms synthesis latency)
- **Jackie Integration** — WebSocket bridge for Android robot communication (audio, video, TTS, DOA)
- **Session Management** — Auto-timeout, farewell detection, quick re-engagement

## Architecture

```
┌─────────────────────┐    WebSocket    ┌──────────────────────┐
│   Jackie Robot       │                │   PC (Edge Server)    │
│   (Android/RK3588)   │   audio/video  │                      │
│                      │ ─────────────→ │  ASR (Parakeet/FW)   │
│   4-mic array (CAE)  │                │  Face Tracking        │
│   Camera (2MP)       │   TTS/commands │  ASD (LASER)         │
│   15.6" Touchscreen  │ ←──────────── │  Engagement Detection │
│   Speakers (2x)      │                │  LLM Dialogue         │
│                      │                │  Behavior Trees        │
└─────────────────────┘                │  Piper TTS            │
                                        └──────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended: RTX 3060 Ti or better)
- Webcam + Microphone

### Installation

```bash
git clone https://github.com/harrey401/HRI-smait.git
cd HRI-smait
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
OPENAI_API_KEY=your_key_here
SMAIT_ASR_BACKEND=parakeet_tdt    # or faster_whisper
SMAIT_DEBUG=1
```

### Run (Standalone — webcam + mic)

```bash
source venv/bin/activate
python -m smait.main
```

### Run (Jackie Robot Mode)

```bash
source venv/bin/activate
python run_jackie.py --port 8765
```

Then connect the Jackie Android app to `ws://<your-ip>:8765`.

## Project Structure

```
smait/
├── core/           # Config, events, data structures
├── sensors/        # Audio pipeline, camera, network sources
├── perception/     # ASR, face tracking, ASD, engagement, speaker verification
├── dialogue/       # LLM backends, conversation memory, dialogue manager
├── behavior/       # Behavior trees (py_trees)
├── output/         # TTS engines, audio playback
└── main.py         # Main HRI system entry point
```

## Configuration

Key settings in `.env`:

| Variable | Options | Default |
|----------|---------|---------|
| `SMAIT_ASR_BACKEND` | `parakeet_tdt`, `faster_whisper` | `parakeet_tdt` |
| `SMAIT_ASD_BACKEND` | `laser` | `laser` |
| `SMAIT_DEBUG` | `0`, `1` | `0` |
| `SMAIT_SHOW_VIDEO` | `0`, `1` | `1` |
| `OPENAI_API_KEY` | Your API key | — |

## Robot Personas

Built-in system prompts for different use cases:

- **`jackie`** — SJSU service robot (default)
- **`demo`** — Event demonstration mode
- **`luma`** — General service assistant
- **`concierge`** — Hotel/venue concierge

## Hardware — Jackie Robot

- **CPU:** Rockchip RK3588, Android 12
- **Microphone:** Linear 4-mic array with iFLYTEK CAE SDK (AEC, beamforming, DOA)
- **Speakers:** 2x loudspeakers
- **Camera:** 2MP face recognition (0.5-3m range)
- **Display:** 15.6" 1080p touchscreen
- **Mobility:** Differential + mecanum wheels, LIDAR SLAM navigation

## Tech Stack

- **ASR:** NVIDIA NeMo Parakeet TDT / OpenAI Whisper
- **Vision:** MediaPipe, OpenCV
- **ASD:** LASER (Landmark-Aware Speaker Estimation in Real-time)
- **LLM:** OpenAI GPT-4o-mini / Ollama (local)
- **TTS:** Piper (local, ~70ms)
- **Behavior:** py_trees
- **Communication:** WebSocket (asyncio)

## Authors

- **Harreynish Gowtham** — ME Masters, SJSU
- BioRob Lab, San José State University

## License

Research use only. Contact author for licensing inquiries.
