# SMAIT v3 — HRI System for Jackie Robot

## What This Project Is

Multimodal Human-Robot Interaction system for Jackie, an iFLYTEK humanoid service robot at SJSU. The robot sees faces, detects who's speaking via audio-visual cues, separates their voice from background noise, transcribes speech, generates responses, and speaks back — all in real-time on a local GPU.

**Architecture:** Android app on Jackie (audio/video capture + TTS playback) ↔ WebSocket ↔ Python server (ML pipeline on RTX 5070)

## Repos

- **This repo** (`HRI-smait`, branch `v3`): Python server — ML models, event system, WebSocket server
- **smait-jackie-app** (`~/projects/smait-jackie-app`): Android app — CAE beamforming, camera, TTS playback

## Key Entry Points

- `run_jackie.py` — Launch the full system: `python run_jackie.py --debug`
- `smait/main.py` — `HRISystem` class: wires all modules, runs async loops
- `smait/core/config.py` — All configuration dataclasses
- `smait/core/events.py` — EventBus (pub/sub between modules)
- `smait/connection/protocol.py` — Binary frame types + JSON message schemas

## ML Pipeline (processing order)

1. **Silero VAD** (`sensors/audio_pipeline.py`) — Speech detection from CAE audio, CPU
2. **Dolphin AV-TSE** (`perception/dolphin_separator.py`) — Target speaker extraction using lip video + audio, GPU ~2.5GB
3. **Parakeet TDT** (`perception/asr.py`) — NVIDIA NeMo ASR, GPU ~2GB. Needs `NEMO_DISABLE_CUDA_GRAPHS=1` for Blackwell
4. **Kokoro-82M** (`output/tts.py`) — Sentence-streaming TTS via KPipeline, GPU ~1GB
5. **L2CS-Net** (`perception/gaze.py`) — Gaze estimation for engagement detection, GPU ~0.3GB
6. **Phi-4 Mini** (`dialogue/manager.py`) — Local LLM via Ollama, with OpenAI API fallback

## Protocol (Jackie ↔ Server)

Binary frames: `[type_byte][payload]`
- `0x01` CAE audio (Jackie→PC): PCM16 mono 16kHz
- `0x02` Video (Jackie→PC): JPEG 640x480
- `0x03` Raw audio (Jackie→PC): PCM16 4-channel 16kHz interleaved
- `0x05` TTS audio (PC→Jackie): PCM16 mono 24kHz

JSON messages: DOA angles, state updates, transcripts, tts_control, cae_status

## Project Status

- **Phases 1-6 (HOME): COMPLETE** — All code written, 119+ unit tests passing
- **Phases 7-8 (LAB): PENDING** — GPU validation, hardware integration, E2E testing
- See `LAB_PLAN.md` for the full lab day implementation guide
- See `.planning/ROADMAP.md` for phase details
- See `.planning/REQUIREMENTS.md` for requirement traceability

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v            # Unit tests (mocked models)
python scripts/smoke_all_models.py    # GPU smoke tests (lab only)
python scripts/test_e2e_pipeline.py   # Full pipeline test (lab only)
bash scripts/lab_runbook.sh           # Everything in order (lab only)
```

## Known Risks

1. **CAE audio format unverified** — assumed PCM16 mono 16kHz from iFLYTEK SDK. First lab test must validate.
2. **Dolphin real-time performance** — designed for 4s offline segments. If too slow, fallback to CAE passthrough is coded.
3. **PyTorch + NeMo + Blackwell** — NeMo may downgrade torch. `setup_lab.sh` checks after every install.

## Development Rules

- Server uses `asyncio` event-driven architecture — modules communicate via `EventBus`
- All ML models have graceful fallbacks (import error → warning, not crash)
- Audio: 16kHz for input processing, 24kHz for TTS output
- Config: `smait/core/config.py` dataclasses, overridable via env vars (`SMAIT_*`) or JSON file
- Android app: single WebSocket connection to server, binary frames for media, JSON for control
