# SMAIT HRI Interview Guide — Learn This Fast

## 90-Second Elevator Pitch

"SMAIT is a multimodal Human-Robot Interaction system I built for my masters project. It runs on Jackie, a physical social robot with a touchscreen, camera, and 4-mic array. The core problem: how do you make a robot that can hold a natural conversation with people walking up to it in a public space?

The system fuses audio and vision in real-time. On the audio side, I use Silero VAD for speech detection, then NVIDIA's Parakeet TDT model for streaming ASR — sub-100ms latency, which is critical for natural turn-taking. On the vision side, I run face detection with MediaPipe, multi-face tracking with DeepSORT, and Active Speaker Detection using LASER — a landmark-assisted approach that tells me which face is actually speaking.

The key technical challenge was audio-visual synchronization. Audio and video arrive at different rates, so I built a temporal buffer that correlates speech segments with ASD results over a 5-second window to correctly attribute who said what.

The dialogue runs through GPT-4o-mini with conversation memory, and responses go back through local TTS on the robot. The whole session lifecycle — approach, engage, converse, disengage — is managed by a behavior tree running at 30Hz."

## Architecture

```
                    Jackie Robot (RK3588 Android)
                   /                              \
          4-mic array                         Camera
          (CAE SDK:                        (WebSocket
          AEC, beamforming,                 → PC)
          noise suppression)                    |
                |                               |
            WebSocket                      VideoSource
                |                               |
         AudioPipeline                    FaceTracker
         (Silero VAD)                    (MediaPipe +
                |                        DeepSORT)
         Parakeet TDT                        |
         (NeMo ASR,                   ActiveSpeakerDetector
          streaming)                  (LASER: landmark-
                |                     assisted, 93% acc)
         SemanticVAD                        |
         (turn prediction)          ASD History Buffer
                \                    (5s temporal window)
                 \                      /
                  SpeakerVerifier ←────┘
                  (fuse audio + vision,
                   verify WHO is speaking)
                        |
                  ASR Confidence Gate
                  (<0.40 + short → reject)
                        |
                  DialogueManager
                  (GPT-4o-mini, memory)
                        |
                  TTS (Piper local /
                       EdgeTTS cloud)
                        |
                  Jackie app → speaker
```

## What Each Module Does

| Module | One-Liner |
|--------|-----------|
| **AudioPipeline** | Silero VAD detects speech, generates segments from ring buffer |
| **Parakeet TDT** | NVIDIA's streaming ASR — 0.6B params, <100ms latency, SOTA English accuracy |
| **SemanticVAD** | Predicts turn boundaries (when speaker is done talking) using pause patterns |
| **FaceTracker** | MediaPipe face detection + DeepSORT for persistent multi-face tracking |
| **LASER ASD** | Landmark-Assisted Speaker Estimation — tells you which tracked face is talking |
| **SpeakerVerifier** | Fuses audio + vision: matches speech to a specific tracked face using temporal ASD buffer |
| **ASR Confidence Gate** | Rejects garbled noise: short utterances with <0.40 confidence get "could you repeat that?" |
| **DialogueManager** | GPT-4o-mini with 10-turn conversation memory, system prompt, async response generation |
| **TTS** | Piper (local, fast, no internet) with EdgeTTS fallback (cloud, more natural) |
| **Behavior Tree** | py_trees at 30Hz: manages session lifecycle — idle → greeting → engaged → farewell |
| **HRISystem** | Main orchestrator: all async tasks, session state machine, crash recovery (10 retries) |

## Design Decisions (Interview Gold)

| Decision | What | Why |
|----------|------|-----|
| **Parakeet over Whisper** | Streaming ASR vs chunk-based | Sub-100ms latency. Whisper chunks = 500ms+ delay = unnatural conversation. |
| **LASER over Light-ASD** | Landmark-based ASD vs deep learning ASD | 93% accuracy, faster inference. Uses MediaPipe landmarks already available. |
| **Temporal ASD buffer** | 5s rolling window for audio-visual sync | Audio and video arrive at different rates. Can't ask "who's talking NOW?" — must correlate over time. |
| **Piper over cloud TTS** | Local TTS primary | Demo can't depend on internet. Piper runs locally, EdgeTTS is fallback. |
| **ASR confidence gate** | Reject low-confidence short utterances | Crowd noise triggers VAD → garbled ASR → nonsense responses. Gate prevents this. |
| **30s session timeout** | Was 45s, reduced | Public demo context — people come and go. 45s holding a dead session felt broken. |
| **Hardware beamforming** | CAE SDK on Jackie, not software filtering | Software DOA filtering was too strict — rejected valid speakers at angles. Trust the hardware. |
| **No personal names in system prompt** | Robot is the face, not the researcher | More professional. "A robotics master's student at SJSU" is enough context. |
| **py_trees behavior tree** | Over raw asyncio state flags | Clean parallel state management. Greeting + monitoring + response generation run simultaneously. |

## Key Technical Challenges (and How You Solved Them)

### 1. Audio-Visual Synchronization
**Problem:** Audio arrives before/after corresponding video frames. Naive approach ("is this face talking right now?") fails because the timings don't align.
**Solution:** 5-second temporal buffer stores ASD results. When a speech segment arrives with a timestamp, the system checks who was speaking DURING that audio window, not at one instant.

### 2. Noisy Environment ASR
**Problem:** Public demo space = crowd noise → VAD triggers on background chatter → garbled ASR → robot talks to nobody.
**Solution:** Multi-layer filtering: (1) Hardware beamforming in Jackie's CAE SDK, (2) Silero VAD with 0.5 threshold, (3) ASR confidence gate rejects short + low-confidence utterances, (4) ASD verifies a visible face is actually speaking.

### 3. Session Lifecycle in Public Spaces
**Problem:** People walk up, talk, walk away mid-sentence, come back. Unlike a chatbot, you can't assume the user stays.
**Solution:** State machine: idle → engaged → farewell with tuned timeouts. 8s face-lost grace (looking away ≠ leaving), 20s reacquisition window (going to grab coffee and coming back), 30s silence timeout.

### 4. NeMo/CUDA Compatibility
**Problem:** NeMo 2.0 had bugs — tuple return format broke Parakeet, CUDA graphs incompatible with Blackwell architecture.
**Solution:** Patched the NeMo wrapper to handle tuple returns and disabled CUDA graph optimization. Filed upstream.

## Common Interview Questions

**Q: "Walk me through what happens when someone walks up and says hello."**
A: Jackie's camera detects a face via MediaPipe → DeepSORT assigns a tracking ID → as they get closer, the behavior tree transitions from IDLE to GREETING. When they speak, the 4-mic array captures audio → CAE SDK does beamforming + noise suppression → audio streams to PC via WebSocket → Silero VAD detects speech → Parakeet TDT transcribes in <100ms → LASER ASD confirms THAT face is the active speaker → SpeakerVerifier matches the speech to the tracked face → DialogueManager sends transcript + history to GPT-4o-mini → response comes back → Piper TTS synthesizes locally → audio plays on Jackie's speaker. The whole loop is ~300-500ms end-to-end.

**Q: "Why not just use ChatGPT's voice mode?"**
A: Three reasons: (1) We need multi-modal fusion — knowing WHO is speaking via face tracking + ASD, not just WHAT they said. (2) We need to run on a physical robot with session lifecycle management. (3) We need low latency for natural interaction — cloud round-trips add 200-500ms per hop. Our pipeline keeps ASR and TTS local.

**Q: "How do you handle multiple people?"**
A: DeepSORT tracks multiple faces simultaneously with persistent IDs. LASER ASD identifies which face is the active speaker. The behavior tree manages primary speaker focus — if a new person starts talking while someone else has the session, it checks ASD to determine if there's a speaker change. Session state is tied to a face ID, not a global "someone is talking" flag.

**Q: "What would you improve?"**
A: (1) Emotion recognition — MediaPipe already gives landmarks, adding expression classification would let Jackie adapt tone. (2) Gesture recognition for non-verbal commands. (3) Better turn-taking with a learned model instead of my pattern-based SemanticVAD. (4) On-device ASR if we upgrade Jackie's SoC — removes PC dependency entirely.

**Q: "How did you evaluate the system?"**
A: Quantitative: ASR word error rate with Parakeet on our test set, ASD accuracy (93% with LASER), end-to-end response latency measurements. Qualitative: user studies during lab walkthroughs — people rated naturalness of conversation, turn-taking smoothness, and overall engagement on a Likert scale. The March 2 demo is the final public evaluation.

## What Makes This Project Impressive

1. **Real hardware, real people** — Not a simulation. A physical robot interacting with humans in a public demo.
2. **Multi-modal fusion** — Audio + vision synchronized in real-time. Most HRI projects do one or the other.
3. **Production engineering** — Crash recovery, confidence gating, graceful degradation (TTS fallback), tuned timeouts. Not a research prototype that breaks.
4. **Full stack** — Hardware (Jackie + mic array) → perception (ASR + face tracking + ASD) → reasoning (LLM + behavior tree) → output (TTS). End to end.
5. **Latency-conscious design** — Every component chosen for speed: streaming ASR, local TTS, landmark-based ASD. Sub-500ms end-to-end.

## Quick Reference

- **Robot:** Jackie (RK3588, Android 12, 15.6" touchscreen, 4-mic iFLYTEK CAE SDK)
- **PC:** RTX 5070, Ubuntu, conda env "smait", Python 3.10
- **ASR:** Parakeet TDT 0.6B (NeMo, streaming, <100ms)
- **ASD:** LASER (landmark-assisted, 93% accuracy)
- **Face Tracking:** MediaPipe + DeepSORT
- **LLM:** GPT-4o-mini (temperature 0.7, 10-turn memory, max 150 tokens)
- **TTS:** Piper (local primary) + EdgeTTS (cloud fallback)
- **Behavior:** py_trees at 30Hz
- **Demo:** March 2, 2026
