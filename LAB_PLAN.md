# SMAIT v3 — Lab Day Implementation Plan

**Date:** Written 2026-03-11
**Goal:** Go from "code-complete at home" to "working conversation on Jackie" in one lab day.
**Estimated time:** 7-10 hours (with debugging buffer)

---

## Pre-Lab Checklist (do at home before going)

- [x] All 12 Python modules implemented (~2,900 LOC)
- [x] 119 unit tests passing, 4 skipped (optional deps)
- [x] Android CaeAudioManager + TtsAudioPlayer written + bugs fixed
- [x] All smoke test scripts written and ready
- [x] Test audio fixtures generated
- [x] setup_lab.sh environment installer ready
- [x] lab_runbook.sh master validation script ready
- [x] Protocol audit: all binary/JSON formats match between Android and server
- [x] AudioTrack restart bug fixed
- [x] Double-speak bug fixed
- [x] tts_state self-conversation loop fixed
- [x] 1008 reconnect loop fixed
- [ ] Record real speech WAV for better ASR testing: `arecord -f S16_LE -r 16000 -c 1 -d 5 scripts/test_audio/sample_16k.wav`
- [ ] Bring OpenAI API key (for LLM fallback if Ollama not set up)
- [ ] Bring Jackie robot charged and ready

---

## Block 1: Environment Setup (1 hour)

### 1.1 Lab PC Setup
```bash
cd ~/projects/SMAIT-v3   # or wherever repo is cloned in lab
git pull origin v3
bash scripts/setup_lab.sh
```

This installs (in order, with version checks):
1. PyTorch 2.7 + CUDA 12.8 (cu128)
2. NeMo toolkit (Parakeet ASR) — **immediately verify torch not downgraded**
3. Kokoro TTS + espeak-ng
4. L2CS-Net (gaze)
5. All remaining deps from requirements.txt

### 1.2 Verify NVIDIA Driver
```bash
nvidia-smi
```
- Need: driver R570+, CUDA 12.8+, RTX 5070 detected
- If driver is old: `sudo apt install nvidia-driver-570` (may need reboot)

### 1.3 Install JDK for Android builds
```bash
sudo apt install openjdk-17-jdk
javac -version  # should show 17+
```

### 1.4 Install Ollama for local LLM
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi4-mini
ollama list  # verify phi4-mini is listed
```
If Ollama not available, set `OPENAI_API_KEY` env var for API fallback.

---

## Block 2: GPU Smoke Tests (1 hour)

Run each individually so failures are isolated:

```bash
source venv/bin/activate

python scripts/smoke_torch.py          # GPU, CUDA, sm_120, tensor ops
python scripts/smoke_vad.py            # Silero VAD on CPU
python scripts/smoke_parakeet.py       # Parakeet ASR transcription
python scripts/smoke_dolphin.py        # Dolphin forward pass — HIGHEST RISK
python scripts/smoke_tts.py            # Kokoro TTS synthesis
python scripts/smoke_gaze.py           # L2CS-Net inference
```

### Known Risk: Dolphin Forward Pass
The Dolphin smoke test uses tensor shapes `audio=[1, samples]` and `video=[1, 1, T, 88, 88, 1]`. If the forward pass fails:
- Try swapping video dims: `[1, 1, T, 1, 88, 88]` or `[1, T, 88, 88, 1]`
- Check the actual model's expected input in `look2hear/models/`
- If nothing works, the system falls back to CAE passthrough (already coded)

### Known Risk: NeMo + PyTorch Version
After running smoke_parakeet.py, verify:
```bash
python -c "import torch; print(torch.__version__)"  # must be 2.7.x
```
If NeMo downgraded torch, reinstall:
```bash
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## Block 3: Combined Model Loading + VRAM Budget (30 min)

```bash
python scripts/smoke_all_models.py
```

Expected output:
```
Model                 Cumulative MB
--------------------  ---------------
baseline                        0 MB
silero_vad                      0 MB
dolphin                     ~2500 MB
parakeet                    ~4500 MB
kokoro                      ~5500 MB
l2cs                        ~5800 MB
PEAK ALLOCATED              ~6500 MB
GPU TOTAL                   12288 MB
HEADROOM                    ~5800 MB
```

If VRAM is tight (>85%):
- Move Kokoro to FP16: saves ~500 MB
- Silero VAD is already on CPU
- L2CS-Net can be moved to CPU (~300 MB saved)

---

## Block 4: E2E Pipeline Test — No Robot (30 min)

```bash
python scripts/test_e2e_pipeline.py
```

This runs the full chain on GPU with synthetic audio:
VAD → Dolphin separation → Parakeet ASR → Kokoro TTS

Key metric: **E2E latency < 1500ms** (separation + ASR + TTS TTFB)

If latency is over target:
- Dolphin: reduce audio segment length (2s → 1s)
- Parakeet: try batch_size=1 explicitly
- Kokoro: already streams by sentence, TTFB should be <100ms

---

## Block 5: Android App Build + Test (30 min)

### 5.1 Build the app
```bash
cd ~/projects/smait-jackie-app
git pull origin main
./gradlew compileDebugKotlin
```

Fix any compilation errors (likely minor SDK path issues).

### 5.2 Run unit tests
```bash
./gradlew test
```

Expect 29 tests (18 CaeAudioManager + 11 TtsAudioPlayer). Fix any failures.

### 5.3 Deploy to Jackie
```bash
./gradlew installDebug
```

Or use Android Studio to build and deploy the APK to the robot.

---

## Block 6: Server Launch + Jackie Connection (1 hour)

### 6.1 Start the server
```bash
cd ~/projects/SMAIT-v3
source venv/bin/activate
python run_jackie.py --debug
```

Note the IP address printed at startup: `Connect Jackie to: ws://<IP>:8765`

### 6.2 Configure Jackie
On the robot's app settings:
- Server IP: the IP from step 6.1
- Server port: 8765
- Tap "Connect"

### 6.3 First Connection Validation
Watch the server logs for:
```
Jackie connected from <android-ip>
```

Then verify data is flowing:
- `SPEECH_DETECTED` events (CAE audio arriving via 0x01)
- `DOA_UPDATE` events (DOA JSON arriving)
- `FACE_UPDATED` events (video frames arriving via 0x02)

### 6.4 CRITICAL: Verify CAE Audio Format
This is the #1 unknown. The CAE SDK's `onAudio` output format has never been verified. Check the server logs:

```
VAD: speech start
VAD: speech segment 1.23s (cae=19680 samples, raw=78720 samples)
```

- `cae` samples should make sense: 1s of 16kHz mono = 16,000 samples
- If the numbers are wildly off (e.g., 8x expected), the CAE is outputting at 48kHz or different format
- If VAD never triggers: the audio format is likely wrong. Add debug logging:
  ```python
  # In AudioPipeline.process_cae_audio(), add after line 217:
  logger.debug("CAE chunk: %d bytes, min=%.4f max=%.4f", len(data), samples.min(), samples.max())
  ```
  Values should be in [-1.0, 1.0] range. If they're all near 0 or all maxed out, format is wrong.

### 6.5 If CAE Audio Format is Wrong
Most likely scenarios and fixes:
- **48kHz instead of 16kHz**: resample with `torchaudio.functional.resample()`
- **PCM32 instead of PCM16**: change `dtype=np.int16` to `dtype=np.int32` and adjust divisor
- **Stereo instead of mono**: reshape and take channel 0

---

## Block 7: Full Conversation Test (1.5 hours)

### 7.1 Voice-Only Mode First
Start without vision to isolate audio issues:
```bash
python run_jackie.py --voice-only --debug
```

Test sequence:
1. Speak to Jackie: "Hello, how are you?"
2. Watch logs for: SPEECH_SEGMENT → SPEECH_SEPARATED → ASR transcript → LLM response → TTS audio
3. Listen for Jackie's audio response through speaker
4. Check for self-echo: does VAD trigger during TTS playback? (should be suppressed by AEC/barge-in logic)

### 7.2 Full Mode (Audio + Vision)
```bash
python run_jackie.py --debug --show-video
```

Test the engagement flow:
1. Walk up to Jackie from ~3m away
2. Look at the robot for >2 seconds (triggers engagement)
3. Jackie should greet you proactively
4. Have a conversation (3-4 turns)
5. Say "goodbye" — session should end gracefully
6. Walk away — should return to idle

### 7.3 Multi-Speaker Test
1. Have two people stand near Jackie
2. One person looks at Jackie and speaks → should be target speaker
3. Other person speaks → Dolphin should separate out their voice
4. Switch who's looking at Jackie → target should switch

### 7.4 Barge-In Test
1. Ask Jackie a long question that will generate a long response
2. While Jackie is speaking, say something
3. Jackie should stop talking (TTS cancelled) and listen to you
4. Jackie should respond to your interruption

---

## Block 8: Latency Measurement + Tuning (1 hour)

### 8.1 Measure E2E Latency
Add timing to the server logs (already built into MetricsTracker):
```
Separation: 250ms, ASR: 180ms, LLM TTFT: 300ms, TTS TTFB: 80ms
Total speech-end to TTS-start: 810ms
```

Target: **< 1500ms total**

### 8.2 Tuning Levers (if over target)

| Component | Current | Tuning Option | Savings |
|-----------|---------|---------------|---------|
| Dolphin | ~200-500ms | Reduce segment to 1s | ~100ms |
| Parakeet | ~100-300ms | Already optimal | — |
| LLM (Ollama) | ~200-500ms | Use smaller quantization | ~100ms |
| LLM (API) | ~300-800ms | Switch to GPT-4o-mini | variable |
| Kokoro TTS | ~50-100ms TTFB | Already streaming | — |

### 8.3 If Dolphin is Too Slow
Dolphin was designed for 4-second offline segments. If real-time latency is >500ms:
1. Try shorter segments (1-2s instead of 4s)
2. If still too slow, disable Dolphin and use CAE passthrough:
   ```python
   # In config: separation.fallback_to_cae = True
   # DolphinSeparator will passthrough if model is unavailable
   ```
   CAE beamforming alone provides decent speaker isolation.

---

## Block 9: Cleanup + Final Validation (1 hour)

### 9.1 Run Full Test Suite
```bash
cd ~/projects/SMAIT-v3
python -m pytest tests/ -v --cov=smait --cov-report=term-missing
```
Target: 80%+ coverage.

### 9.2 Run Android Tests
```bash
cd ~/projects/smait-jackie-app
./gradlew test
```

### 9.3 Dead Code Cleanup
Quick scan for obvious dead code:
```bash
# Unused imports
ruff check smait/ --select F401
# Unused variables
ruff check smait/ --select F841
```

### 9.4 Final Demo Recording
Record a successful conversation for the HFES presentation:
- Start fresh session
- Walk up, get greeted
- 3-4 turn conversation
- Say goodbye
- Capture server logs: `cp smait.log logs/demo_$(date +%Y%m%d).log`

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No audio events in server log | CAE not initialized / token expired | Check CaeCoreHelper.java token, verify CAE .so loaded |
| VAD never triggers | Audio format mismatch (not PCM16 16kHz) | Log raw bytes, check format |
| ASR outputs garbage | Wrong sample rate or encoding | Verify with `smoke_parakeet.py` on real audio |
| TTS plays once then silent | AudioTrack not restarted | Already fixed — verify ensurePlaying() deployed |
| Robot talks to itself | tts_state not handled | Already fixed — verify manager.py deployed |
| Dolphin crashes on GPU | Tensor shape mismatch | Try different dim ordering, check model docs |
| VRAM OOM | Too many models loaded | Move L2CS to CPU, use FP16 for Kokoro |
| Jackie can't connect | Wrong IP, firewall, or port | Check `ip addr`, try `--host 0.0.0.0` |
| Double reconnect loop | Stale connection + 1008 rejection | Already fixed — verify 15s delay deployed |
| LLM not responding | Ollama not running or wrong model | `ollama list`, `ollama pull phi4-mini` |
| No video frames | Camera permission denied on Jackie | Check Android app permissions |
| Engagement never triggers | Gaze estimation not running | Check L2CS loaded, try `--show-video` to see overlay |

---

## Success Criteria

The lab day is DONE when:
1. All smoke tests pass (Block 2-4)
2. Jackie connects and streams all 3 data types (Block 6)
3. At least one full conversation completes end-to-end (Block 7)
4. E2E latency is under 1500ms (Block 8)
5. Unit tests pass on both repos (Block 9)

## Files Changed Today (for reference)

### SMAIT-v3 (server)
- `scripts/` — 12 new scripts (smoke tests, setup, E2E, fixtures)
- `scripts/test_audio/` — 4 WAV fixtures
- `tests/unit/test_gaze.py` — skipif for missing l2cs
- `tests/unit/test_tts.py` — skipif for missing kokoro
- `smait/connection/manager.py` — tts_state handler (self-conversation fix)

### smait-jackie-app (Android)
- `TtsAudioPlayer.kt` — ensurePlaying() + pause instead of stop
- `MainActivity.kt` — remove double-speak, handle tts_control:start, 1008 reconnect delay
