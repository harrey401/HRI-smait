# SMAIT HRI System v2.0

**Multimodal Human-Robot Interaction System for Service Robots**

A robust, hardware-agnostic framework for natural human-robot conversation that works across real robots, simulation, and development environments.

## Key Features

- **Ultra-Low-Latency Speech Recognition**: Parakeet TDT (NVIDIA NeMo) with streaming inference (~100ms)
- **LASER Active Speaker Detection**: Lip-landmark assisted detection for robust multi-speaker environments
- **Engagement Detection**: Proximity, attention, and greeting-based session management
- **Semantic Turn-Taking**: Pattern-based end-of-turn prediction for faster responses
- **Behavior Trees**: py_trees architecture for composable, parallel behaviors
- **Voice Activity Detection**: Silero VAD for accurate speech boundary detection
- **Piper TTS**: Fast local neural TTS (~100ms) with Edge TTS fallback
- **Multi-Platform Support**: Standalone, ROS 2, and Isaac Sim
- **Sliding Window Memory**: Efficient conversation context management


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smait-hri.git
cd smait-hri

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install PyTorch with CUDA (recommended for fast ASR)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# First run will download Parakeet TDT model (~1.2GB)
```

### Configuration

Create a `.env` file:

```bash
# LLM Configuration (required)
OPENAI_API_KEY=your_openai_api_key

# ASR Configuration
SMAIT_ASR_BACKEND=parakeet_tdt    # Recommended (or faster_whisper as fallback)
PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2

# Speaker Detection
SMAIT_ASD_BACKEND=laser           # Recommended

# Hardware
CAMERA_INDEX=0

# Debug
SMAIT_DEBUG=1
SMAIT_SHOW_VIDEO=1
```

### Run

```bash
# Run the main system
smait

# Or run directly
python -m smait.main

# Use faster-whisper fallback (if NeMo not installed)
SMAIT_ASR_BACKEND=faster_whisper python -m smait.main
```


## Architecture

```
+------------------------------------------------------------------+
|                      SMAIT HRI System v2.0                       |
+------------------------------------------------------------------+
|                                                                  |
|  +-------------+     +-------------+     +-------------+         |
|  |   Sensors   |     | Perception  |     |  Dialogue   |         |
|  |             |     |             |     |             |         |
|  | - Camera    |---->| - LASER ASD |---->| - Memory    |         |
|  | - Microphone|     | - Parakeet  |     | - OpenAI    |         |
|  | - (Isaac)   |     | - Semantic  |     | - TTS       |         |
|  +-------------+     |   VAD       |     +-------------+         |
|         |            +-------------+            |                |
|         +-------------------+-------------------+                |
|                             v                                    |
|                    +------------------+                          |
|                    |  Behavior Tree   |                          |
|                    |   (py_trees)     |                          |
|                    |                  |                          |
|                    | - TrackFaces     |                          |
|                    | - Listen         |                          |
|                    | - Backchannel    |                          |
|                    | - Respond        |                          |
|                    +------------------+                          |
|                                                                  |
+------------------------------------------------------------------+
```

## Project Structure

```
smait_hri_v2/
├── smait/
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   └── events.py           # Data types and events
│   ├── sensors/
│   │   ├── sources.py          # Camera/mic abstraction
│   │   └── audio_pipeline.py   # VAD + audio processing
│   ├── perception/
│   │   ├── transcriber.py      # ASR manager (Parakeet/Whisper)
│   │   ├── parakeet_asr.py     # Parakeet TDT engine
│   │   ├── face_tracker.py     # MediaPipe face tracking
│   │   ├── asd.py              # ASD manager
│   │   ├── laser_asd.py        # LASER backend
│   │   ├── verifier.py         # Speaker verification
│   │   ├── engagement.py       # Engagement + farewell detection
│   │   └── semantic_vad.py     # Turn-taking prediction
│   ├── output/
│   │   └── tts.py              # TTS engines (Piper, Edge)
│   ├── behavior/
│   │   ├── tree.py             # Main behavior tree
│   │   └── behaviors.py        # Behavior implementations
│   ├── dialogue/
│   │   └── manager.py          # LLM + memory
│   └── main.py                 # Entry point
├── ros2_ws/                    # ROS 2 packages
├── docs/                       # Documentation
├── tests/                      # Unit tests
├── requirements.txt
└── pyproject.toml
```


## Speech Detection Flow

The system uses temporal buffering to match audio with visual speech detection:

```
1. VAD TRIGGER
   └─> Silero VAD detects voice activity → mark_speech_start()

2. DURING SPEECH
   └─> Audio buffered in ring buffer
   └─> ASD runs on each video frame, results stored in 5-second history

3. SPEECH END
   └─> VAD detects 500ms silence → mark_speech_end()
   └─> Audio sent to Parakeet TDT for transcription

4. VERIFICATION
   └─> Query ASD history: "Who was speaking during [start, end]?"
   └─> Calculate speaking score per face
   └─> New session: check engagement (proximity + attention + greeting)
   └─> Active session: verify same user, check for farewell
   └─> REJECT if: no visual speech OR different speaker OR not engaged

5. ENGAGEMENT (new sessions)
   └─> Proximity: face close enough (bbox area >= 15000px)
   └─> Attention: facing robot (head yaw < 35 degrees)
   └─> Greeting: must say hello/hi/hey/etc. to start
   └─> Farewell: bye/goodbye ends session after response

6. RESPONSE
   └─> Accepted speech → LLM → TTS → audio response
```

This temporal approach handles audio-visual delay and ensures only the target user's speech triggers responses.


## ASR Backend Options

### Parakeet TDT (Recommended)
```bash
SMAIT_ASR_BACKEND=parakeet_tdt
PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2
```
- **Latency**: <100ms (streaming)
- **Accuracy**: State-of-the-art on Open ASR Leaderboard
- **License**: CC-BY-4.0 (open source)
- **Requirements**: NVIDIA GPU, NeMo toolkit

### faster-whisper (Fallback)
```bash
SMAIT_ASR_BACKEND=faster_whisper
WHISPER_MODEL=base.en
```
- **Latency**: ~300-800ms (batch)
- **Accuracy**: Good
- **Requirements**: CPU or NVIDIA GPU


## Speaker Detection Options

### LASER (Recommended)
```bash
SMAIT_ASD_BACKEND=laser
```
- Uses MediaPipe lip landmarks (already computed)
- Robust to audio-visual desync
- Works in lite mode without model file

### MAR Heuristic (Fallback)
```bash
SMAIT_ASD_BACKEND=mar_heuristic
```
- Geometric mouth aspect ratio
- Motion-compensated
- No external model needed


## TTS Options

### Piper TTS (Recommended)
- **Latency**: ~100-160ms (local)
- **Quality**: Good neural voice
- **Requirements**: Auto-downloads ~50MB model
- Voice models stored in `~/.smait/models/piper/`

### Edge TTS (Fallback)
- **Latency**: ~1200ms (cloud)
- **Quality**: Excellent (Microsoft neural voices)
- **Requirements**: Internet connection


## Performance (with CUDA)

| Metric | v1.0 | v2.0 Target | v2.0 Actual |
|--------|------|-------------|-------------|
| ASR Latency | 1-2s | <200ms | ~64ms (CUDA) |
| TTS Latency | ~1.2s | <200ms | ~100ms (Piper) |
| Total Response | 2-4s | <500ms | ~800ms |
| ASD per frame | N/A | <20ms | <1ms (LASER-lite) |
| ASD Accuracy | ~85% | >90% | ~93% |

Note: ~600ms of response time is the OpenAI API call. Local processing (ASR + ASD + TTS) is very fast.

### CUDA Setup

For optimal performance, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```


## References

- [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) - NVIDIA's state-of-the-art ASR
- [LASER Paper](https://arxiv.org/abs/2501.11899) - Lip Landmark Assisted Speaker Detection
- [py_trees](https://py-trees.readthedocs.io/) - Behavior Trees for Python
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Fast Whisper inference
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection


## Acknowledgments

- NVIDIA NeMo and Isaac Sim teams
- LASER authors (UC Davis)
- py_trees contributors


## License

MIT License
