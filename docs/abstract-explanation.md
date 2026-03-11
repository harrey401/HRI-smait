# SMAIT v2.0 — Architecture Explanation

## Overview

SMAIT is a distributed system split across two devices: the Jackie robot (sensor and actuator platform) and an edge server PC with an NVIDIA GPU (all AI inference). They communicate over WebSocket on a dedicated 5GHz WiFi network.

## Jackie Robot

Jackie is a mobile service robot built on the Rockchip RK3588 SoC running Android 12. It carries a 4-microphone linear array processed by the iFLYTEK CAE SDK, which performs hardware-level acoustic echo cancellation (AEC), beamforming, noise suppression, and Direction of Arrival (DOA) estimation. A front-facing RGB camera captures video at 640×480 resolution and 30 frames per second. Jackie also has a 15.6-inch touchscreen and built-in speakers for TTS playback. Jackie does not run any AI models — it captures sensor data and streams it to the PC, and receives synthesized speech back for playback.

## Communication

Audio and video frames are streamed from Jackie to the PC over WebSocket using a binary protocol. Audio frames carry a type prefix byte to distinguish between CAE-processed audio and raw multi-channel audio. Video frames are JPEG-compressed. TTS audio and control commands are sent back from the PC to Jackie over the same connection. The system uses a dedicated WiFi network (GL.iNet travel router) because the university's eduroam network blocks device-to-device communication. USB ADB port forwarding serves as a fallback.

## Edge Server — Audio Path

When audio arrives at the PC, it enters the audio processing pipeline. First, Silero VAD (Voice Activity Detection) segments the continuous audio stream into speech regions, filtering out silence and background noise. Only segments identified as containing speech are forwarded to the ASR stage.

The speech segments are transcribed by NVIDIA Parakeet TDT 0.6B, a Token-and-Duration Transducer model from the NeMo toolkit running on the GPU. It achieves sub-100ms transcription latency. Each transcription comes with a confidence score.

The confidence gate checks the ASR output — if the confidence score falls below 0.40 and the utterance is short, the transcription is rejected and the robot asks the user to repeat themselves. This prevents the system from responding to misheard or ambiguous audio.

## Edge Server — Vision Path

In parallel, camera frames from Jackie are processed through the vision pipeline. MediaPipe detects faces in each frame, and DeepSORT assigns persistent tracking IDs across frames so the system can follow individuals over time even as they move.

Tracked faces are passed to the LASER Active Speaker Detection module, which correlates lip movements with the incoming audio signal using a temporal buffer to handle audio-visual synchronization delays. This determines which of the visible faces is actually speaking.

The Speaker Verifier makes the final decision: Accept (this person is speaking to the robot), Reject (speech is coming from a bystander), or NoFace (no face detected). Only accepted speech triggers a response.

## Convergence

The audio and vision paths merge at the Speaker Verifier. A verified transcript — one that has passed both the ASR confidence gate and the speaker verification — is forwarded to the Semantic VAD module. Semantic VAD performs turn-taking prediction, analyzing the transcript content and timing patterns to determine when the speaker has finished their turn. This enables the system to respond at natural pause points rather than cutting the speaker off or waiting too long.

## Dialogue and Response

The verified, complete utterance is sent to the LLM (GPT-4o-mini) along with a sliding window of the last 10 conversation turns for context. The LLM generates a conversational response appropriate to the interaction context.

The response text is synthesized into speech by Piper TTS, a lightweight local text-to-speech engine with approximately 70ms synthesis latency. If Piper fails, the system falls back to Microsoft EdgeTTS (cloud-based). The synthesized audio is sent back to Jackie over WebSocket for playback through the robot's speakers.

## Session Lifecycle

A py_trees behavior tree running at 30Hz manages the interaction lifecycle through five states:

1. **Idle** — Robot is waiting. No active interaction.
2. **Approach** — A person has been detected approaching. Face tracking and gaze estimation monitor whether they intend to interact.
3. **Engaged** — The person is looking at the robot and within interaction range. The robot greets them proactively — no wake word is needed.
4. **Conversation** — Active multi-turn dialogue. The full audio-vision-language pipeline is running.
5. **End** — The interaction has concluded via explicit farewell, 30-second silence timeout, or the person walking away (8-second grace period before ending).

After ending, the system returns to Idle. If the same person returns within 20 seconds, the system re-engages and resumes the previous conversation context rather than starting fresh.
