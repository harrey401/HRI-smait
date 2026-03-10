# Phase 6: Android Audio Pipeline - Research

**Researched:** 2026-03-10
**Domain:** Android Kotlin, iFlytek CAE SDK (JNI), Android AudioTrack, WebSocket binary protocol
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUD-01 | Android app integrates CAE beamforming (merge cae-work-march2 via revert-the-revert) | CaeAudioManager.kt on branch cae-work-march2 is the source of truth; revert-the-revert approach documented below |
| AUD-02 | CAE 8-channel to 4-channel format mismatch resolved via hlw.ini config | Channel adapter function `adapeter4Mic32bit` is the correct pattern; hlw.ini has no channel count key — fix is in Kotlin adapter code |
| AUD-03 | Android app sends 3 streams: CAE audio (0x01), raw 4-channel audio (0x03), video (0x02) | Current CaeAudioManager only sends 0x01 (bypassed) — must add real CAE output on 0x01 + raw 4ch on 0x03 |
| AUD-04 | Android app sends DOA angles from CAE callbacks as JSON messages | Current CaeAudioManager sends DOA as binary 0x03 frame — must change to JSON text `{"type":"doa","angle":N,"beam":N}` |
| TTS-04 | Android app plays TTS audio via AudioTrack on speaker | No binary WebSocket handler exists in main branch — must add `onMessage(ByteString)` + AudioTrack player |
</phase_requirements>

---

## Summary

Phase 6 is an Android Kotlin coding phase — not a Python server phase. The server-side protocol is already complete (0x01/0x02/0x03/0x05 binary framing, JSON DOA parsing). All work is in the `smait-jackie-app` Android project at `/home/gow/.openclaw/workspace/projects/smait-jackie-app/`.

The CAE SDK beamforming integration lives entirely on the `cae-work-march2` git branch as `CaeAudioManager.kt`. The main branch was reverted to AudioRecord for demo stability. Phase 6 reverses that revert ("revert-the-revert"), then fixes the three gaps left from the original branch work: (1) the channel adapter passes wrong format to CAE, (2) DOA is sent as binary instead of JSON, and (3) raw 4-channel audio stream (0x03) is never sent.

TTS-04 is an independent feature: the app currently plays TTS via Android's TextToSpeech engine receiving JSON `{"type":"tts","text":"..."}`. The server is already sending PCM16 binary 0x05 frames — the app just lacks the `onMessage(ByteString)` handler and an AudioTrack player to receive and play them.

**Primary recommendation:** Two plans. Plan 01 = revert-the-revert + fix CAE channel adapter + fix DOA to JSON + add raw 4ch stream. Plan 02 = add binary WebSocket handler + AudioTrack TTS playback.

---

## Standard Stack

### Core (Android side — already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| OkHttp | 4.12.0 | WebSocket client | Already in use for all WS communication |
| `android.media.AudioTrack` | Android SDK | PCM audio playback | Native Android — no dependency needed |
| `android.media.AudioRecord` | Android SDK | PCM audio capture (fallback) | Being replaced by CAE SDK |
| CAE SDK (`cae.jar`, `AlsaRecorder.jar`) | proprietary | iFlytek beamforming engine | Already in `app/libs/`, JNI wired |
| Kotlin coroutines | stdlib | Background threading | Standard in modern Android |

### Supporting (server side — already complete)
| Library | Version | Purpose | Note |
|---------|---------|---------|------|
| `websockets` | 12.0+ | WS server | Already implemented — no changes needed |
| `smait/connection/protocol.py` | — | Binary frame framing | Complete: 0x01/0x02/0x03/0x04/0x05 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `AudioTrack` | `MediaPlayer` | MediaPlayer wraps files/URIs, not raw PCM streams — AudioTrack is the right choice |
| Binary DOA frame | JSON DOA message | Server expects JSON; binary 0x03 would collide with AUDIO_RAW type — JSON is correct |
| OkHttp `onMessage(ByteString)` | Manual HTTP/WebSocket | OkHttp already handles binary frames via listener override — no extra dependency |

**Installation:** No new dependencies needed. Everything is already in `build.gradle.kts`.

---

## Architecture Patterns

### Recommended Project Structure (Android changes only)
```
smait-jackie-app/app/src/main/java/com/gow/smaitrobot/
├── MainActivity.kt          # Modified: wire CaeAudioManager, add binary WS handler
├── CaeAudioManager.kt       # New file (from cae-work-march2 + fixed)
└── TtsAudioPlayer.kt        # New file: AudioTrack PCM16 playback
```

### Pattern 1: CAE SDK Audio Pipeline (the correct channel adapter)
**What:** ALSA delivers 8ch 16-bit interleaved PCM. CAE expects 6ch 32-bit with embedded channel IDs. The reference adapter (`adapeter4Mic32bit` from `CaeOperator.java` in the CAE demo) uses format `[0x00, channel_id, pcm_lo, pcm_hi]` per sample — channel IDs 1..6, mapping 8ch input to mics 1-4 + ref1 (ch6) + ref2 (ch7).

**When to use:** Every ALSA PCM callback before calling `caeCoreHelper.writeAudio(data)`.

**Example (correct adapter — from CAE demo `CaeOperator.java`):**
```kotlin
// Source: docs/hardware-sdk/CAEDemoAIUI-4 MIC/CaeOperator.java adapeter4Mic32bit()
// Input: 8ch x 2 bytes = 16 bytes/frame (S16_LE interleaved)
// Output: 6ch x 4 bytes = 24 bytes/frame (channel-id-prefixed S16 in 32-bit slot)
private fun adapt8chTo6chCaeFormat(data: ByteArray): ByteArray {
    val frames = data.size / 16
    val out = ByteArray(frames * 24)
    // Channel mapping: ALSA ch0-3 = mic1-4, ch6 = ref1, ch7 = ref2
    val srcOffsets = intArrayOf(0, 2, 4, 6, 12, 14)
    for (j in 0 until frames) {
        val inOff = j * 16
        val outOff = j * 24
        for (ch in 0 until 6) {
            val sOff = inOff + srcOffsets[ch]
            val dOff = outOff + ch * 4
            out[dOff + 0] = 0x00
            out[dOff + 1] = (ch + 1).toByte()  // channel IDs 1..6
            out[dOff + 2] = data[sOff + 0]
            out[dOff + 3] = data[sOff + 1]
        }
    }
    return out
}
```

**CRITICAL DIFFERENCE from cae-work-march2 branch:** The branch adapter used `[0x00, 0x00, pcm_lo, pcm_hi]` (no channel ID). The demo code uses `[0x00, ch_id, pcm_lo, pcm_hi]`. The branch adapter is likely why CAE `onAudio` callback never fired reliably — channel IDs are required.

### Pattern 2: Dual Stream Send (CAE output + raw 4ch)
**What:** CaeAudioManager must send two audio streams per ALSA callback:
- `0x01` (AUDIO_CAE): CAE-processed beamformed mono from `onAudio` callback
- `0x03` (AUDIO_RAW): 4-channel raw audio extracted from 8ch ALSA data

**When to use:** Every ALSA PCM callback and every CAE `onAudio` callback.

**Example:**
```kotlin
// Raw 4ch stream: extract ch0-3 from 8ch ALSA data, send as 0x03
private fun sendRaw4ch(alsaData: ByteArray) {
    val frames = alsaData.size / 16  // 16 bytes per 8ch frame
    val raw4ch = ByteArray(frames * 8)  // 4ch x 2 bytes
    for (j in 0 until frames) {
        System.arraycopy(alsaData, j * 16,     raw4ch, j * 8, 8)  // ch0-3
    }
    val frame = ByteArray(1 + raw4ch.size)
    frame[0] = 0x03  // AUDIO_RAW
    System.arraycopy(raw4ch, 0, frame, 1, raw4ch.size)
    webSocket?.send(frame.toByteString())
}

// CAE beamformed output: send from onAudio callback as 0x01
override fun onAudio(audioData: ByteArray, dataLen: Int) {
    val frame = ByteArray(1 + dataLen)
    frame[0] = 0x01  // AUDIO_CAE
    System.arraycopy(audioData, 0, frame, 1, dataLen)
    webSocket?.send(frame.toByteString(0, frame.size))
}
```

### Pattern 3: DOA as JSON text frame (not binary)
**What:** Server `_handle_text()` parses `{"type":"doa","angle":N,"beam":N}`. The cae-work-march2 branch incorrectly sends DOA as a binary 0x03 frame which would be misinterpreted as AUDIO_RAW.

**Correct implementation:**
```kotlin
private fun sendDoaAngle(angle: Int, beam: Int) {
    val json = JSONObject().apply {
        put("type", "doa")
        put("angle", angle)
        put("beam", beam)
    }
    webSocket?.send(json.toString())
}
```

### Pattern 4: AudioTrack PCM16 playback for 0x05 TTS frames
**What:** Server sends PCM16 mono 24kHz audio as binary WebSocket frames with type byte 0x05. Android must buffer these and play via AudioTrack. OkHttp's `WebSocketListener` has a separate `onMessage(ws, bytes: ByteString)` override for binary frames.

**When to use:** On receipt of binary WS message starting with 0x05.

**Example (AudioTrack setup for Kokoro 24kHz output):**
```kotlin
// Source: Android AudioTrack API docs (developer.android.com/reference/android/media/AudioTrack)
// Kokoro TTS outputs 24kHz PCM16 mono
private val audioTrack = AudioTrack.Builder()
    .setAudioAttributes(AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build())
    .setAudioFormat(AudioFormat.Builder()
        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
        .setSampleRate(24000)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
        .build())
    .setTransferMode(AudioTrack.MODE_STREAM)
    .setBufferSizeInBytes(
        AudioTrack.getMinBufferSize(24000, AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT) * 4
    )
    .build()

// In WebSocketListener:
override fun onMessage(ws: WebSocket, bytes: ByteString) {
    val data = bytes.toByteArray()
    if (data.isEmpty()) return
    when (data[0]) {
        0x05.toByte() -> {
            val pcm = data.sliceArray(1 until data.size)
            ttsAudioPlayer.write(pcm)
        }
    }
}
```

### Pattern 5: Revert-the-Revert via git cherry-pick
**What:** Commit `bc88c6e` on main is the revert of CAE work. The correct approach is NOT to merge `cae-work-march2` directly (it has many debug commits). Instead: cherry-pick the clean `CaeAudioManager.kt` content from the branch tip and apply the fixes.

**Why not direct merge:** The branch has 20+ debug/experiment commits including "bypass CAE: send ch0 mono with 16x gain directly" — these are not production code. We want the final state of CaeAudioManager.kt from branch tip + bug fixes.

**Procedure:**
```bash
# In smait-jackie-app repo:
# Option A: cherry-pick clean files from branch
git show cae-work-march2:app/src/main/java/com/gow/smaitrobot/CaeAudioManager.kt > /tmp/CaeAudioManager.kt
# Then edit to fix channel adapter + DOA format + add raw stream

# Option B: revert the revert commit
git revert bc88c6e --no-commit  # Reverses the "revert: back to AudioRecord" commit
# Then apply fixes on top
```

### Anti-Patterns to Avoid
- **Sending DOA as binary 0x03:** 0x03 is AUDIO_RAW — the server would interpret DOA bytes as raw audio PCM. Always send DOA as JSON text.
- **Using wrong channel adapter:** `[0x00, 0x00, pcm_lo, pcm_hi]` (no channel ID) is what the branch used and why CAE `onAudio` didn't fire. Use `[0x00, ch_id, pcm_lo, pcm_hi]`.
- **Playing TTS via Android TextToSpeech for 0x05 frames:** The server is sending raw PCM16 — Android TTS expects text strings. AudioTrack must handle binary audio.
- **Using `AudioTrack.MODE_STATIC`:** TTS audio arrives in streaming chunks. Use `MODE_STREAM` with continuous `write()` calls.
- **Blocking the WebSocket callback thread:** `AudioTrack.write()` can block. Use a background HandlerThread or Kotlin coroutine for audio playback.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCM16 audio playback | Custom audio decoder | `android.media.AudioTrack` | Native Android handles buffering, sample rate conversion, hardware routing |
| WebSocket binary framing | Manual byte assembly | OkHttp's `ByteString` + `onMessage(ByteString)` | Already wired, handles fragmentation and reassembly |
| ALSA device access | Custom native code | `AlsaRecorder.jar` (already in libs) | Hardware-specific JNI already written for this robot's USB dongle |
| CAE engine lifecycle | Custom audio engine | `CaeCoreHelper.java` + `cae.jar` | Proprietary SDK handles beamforming/AEC/noise suppression |

**Key insight:** All JNI wiring for ALSA and CAE is already done. The work is purely in Kotlin: fix the channel adapter, fix DOA format, add raw stream, add AudioTrack player.

---

## Common Pitfalls

### Pitfall 1: CAE `onAudio` callback never fires
**What goes wrong:** `CaeCoreHelper` is initialized and `writeAudio()` is called, but `onAudioCallback` / `onAudio` is never triggered.
**Why it happens:** Two known causes: (a) channel adapter format wrong — CAE rejects audio without proper channel IDs; (b) beam set to -1 (wake-word gate mode) — `CAESetRealBeam(-1)` blocks output until wake word detected.
**How to avoid:** Use correct channel adapter with IDs 1..6. Call `CAE.CAESetRealBeam(0)` immediately after `CaeCoreHelper` init to enable continuous output.
**Warning signs:** `caeCallbackCount` stays at 0 after 100+ ALSA callbacks.

### Pitfall 2: Auth token failure (`鉴权失败`)
**What goes wrong:** `CaeCoreHelper.EngineInit()` calls `CAE.CAEAuth("38dedb6f-...")` and returns -1.
**Why it happens:** The hardcoded auth token in `CaeCoreHelper.java` is tied to an iFlytek developer account. It may be expired or bound to a specific device.
**How to avoid:** Check logcat for "鉴权失败" during first run in lab. If auth fails, need a valid token from the iFlytek account associated with the hardware.
**Warning signs:** `EngineInit()` returns false; no audio callbacks regardless of channel format.

### Pitfall 3: DOA 0x03 binary frame collision with AUDIO_RAW
**What goes wrong:** Server interprets DOA bytes as audio PCM — logs "AUDIO_RAW received" with garbage content, DOA_UPDATE events never emitted.
**Why it happens:** cae-work-march2 branch sends DOA as binary 0x03 frame (5 bytes: type + float). Server routes 0x03 to `SPEECH_DETECTED` event as raw audio.
**How to avoid:** Always send DOA as JSON text frame. The `WebSocketListener.onMessage(String)` path is for JSON; `onMessage(ByteString)` is for binary.
**Warning signs:** No `DOA_UPDATE` events in server logs; "AUDIO_RAW" logged on DOA angle updates.

### Pitfall 4: AudioTrack sample rate mismatch
**What goes wrong:** Audio playback sounds like chipmunks (too fast) or slow/low (too slow).
**Why it happens:** Kokoro TTS outputs 24kHz PCM16. If AudioTrack is configured for 16kHz or 44.1kHz, playback speed is wrong.
**How to avoid:** Always set `setSampleRate(24000)` to match Kokoro output.
**Warning signs:** Intelligible but wrong-pitched speech, or correct text at wrong speed.

### Pitfall 5: AudioTrack blocked in WebSocket callback
**What goes wrong:** WebSocket stops receiving frames while AudioTrack is playing.
**Why it happens:** OkHttp WebSocket callbacks run on the OkHttp dispatch thread. If `AudioTrack.write()` blocks there, new frames queue up behind it.
**How to avoid:** Use a dedicated HandlerThread or background coroutine scope for `audioTrack.write()`. The WebSocket callback should only enqueue PCM data to a buffer/queue.
**Warning signs:** Audio plays but with dropouts; WebSocket messages arrive late.

### Pitfall 6: ALSA device card number changes after reboot
**What goes wrong:** CAE can't open the USB mic array — "ALSA recording failed to start: -1".
**Why it happens:** ALSA card numbers are assigned dynamically at boot. `PCM_CARD = 2` is correct for the lab hardware but may vary.
**How to avoid:** Document that card number must be verified with `cat /proc/asound/cards` on the robot. Make PCM_CARD configurable (e.g., SharedPreferences or a companion JSON config).
**Warning signs:** `alsaRecorder.startRecording()` returns non-zero; no audio data received at all.

---

## Code Examples

Verified patterns from the codebase and Android SDK:

### CAE Engine Init with Beam Force
```kotlin
// Source: com/voice/osCaeHelper/CaeCoreHelper.java (already in project)
// After CaeCoreHelper() constructor, immediately force beam to bypass wake-word gate:
caeCoreHelper = CaeCoreHelper(caeListener, false)  // false = not 2-mic
com.iflytek.iflyos.cae.CAE.CAESetRealBeam(0)       // 0 = beam 0, -1 = wait for wake word
```

### Binary WebSocket Handler (OkHttp)
```kotlin
// Source: OkHttp WebSocketListener API
// In MainActivity connectToServer():
webSocket = client.newWebSocket(request, object : WebSocketListener() {
    override fun onMessage(ws: WebSocket, text: String) {
        handleServerMessage(text)  // existing JSON handler
    }
    override fun onMessage(ws: WebSocket, bytes: okio.ByteString) {
        handleBinaryMessage(bytes.toByteArray())  // new binary handler
    }
})

private fun handleBinaryMessage(data: ByteArray) {
    if (data.isEmpty()) return
    when (data[0]) {
        0x05.toByte() -> ttsAudioPlayer.write(data, 1, data.size - 1)
        else -> Log.w(TAG, "Unknown binary frame: 0x${data[0].toUByte().toString(16)}")
    }
}
```

### AudioTrack Streaming Setup (24kHz PCM16)
```kotlin
// Source: Android developer docs - AudioTrack streaming mode
private fun createAudioTrack(): AudioTrack {
    val minBuf = AudioTrack.getMinBufferSize(
        24000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
    return AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build())
        .setAudioFormat(AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(24000)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build())
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(minBuf * 4)
        .build().also { it.play() }
}
```

### Raw 4ch Extraction from 8ch ALSA Data
```kotlin
// Source: CaeOperator.java adapeter4Mic() — adapted for 16-bit raw pass-through
// Extract channels 0-3 (mic 1-4) from 8ch 16-bit interleaved
private fun extract4chRaw(alsaData: ByteArray): ByteArray {
    val frames = alsaData.size / 16  // 16 bytes per 8ch frame
    val out = ByteArray(frames * 8)  // 4ch x 2 bytes
    for (j in 0 until frames) {
        System.arraycopy(alsaData, j * 16, out, j * 8, 8)  // ch0-ch3
    }
    return out
}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `AudioRecord` mono 16kHz | CAE SDK via ALSA → 4-mic beamformed mono | Better noise suppression, DOA angles, AEC |
| Android TextToSpeech (text input) | AudioTrack PCM16 playback (0x05 binary frames) | Kokoro voice quality vs Android TTS, no network round-trip |
| DOA binary 0x03 frame (branch bug) | DOA JSON `{"type":"doa","angle":N,"beam":N}` | Server can actually parse it |

**Current state of the branch (cae-work-march2 tip):**
- `CaeAudioManager.kt` exists: good foundation but with three bugs (wrong channel adapter, binary DOA, no raw stream)
- Main branch reverted to AudioRecord: "revert: back to AudioRecord (pre-CAE) for demo - CAE work saved on branch cae-work-march2"
- CAE libs + assets already in main branch (`app/libs/`, `app/src/main/assets/`) — the revert only removed the Kotlin code

---

## Open Questions

1. **Auth token validity in lab**
   - What we know: Token `38dedb6f-de59-4a75-9f24-4ce9cf82e176` is hardcoded in `CaeCoreHelper.java`
   - What's unclear: Whether it's still valid for the lab's hardware (may be expired or device-locked)
   - Recommendation: Cannot verify at home. Plan should include a lab validation step; if auth fails, the iFlytek account owner must generate a new token. Auth failure is detectable immediately in logcat.

2. **CAE output audio format at server**
   - What we know: CAE `onAudio` returns `ByteArray` described as "beamformed mono 16-bit"
   - What's unclear: Is it 16kHz or 48kHz? The ALSA input is 16kHz, but CAE SDK may upsample internally
   - Recommendation: First lab test should log `dataLen / duration` to verify sample rate. Server's `AudioPipeline` expects 16kHz; if CAE outputs 48kHz, downsampling is needed.

3. **Raw 4ch format for Dolphin**
   - What we know: Server expects raw 4-channel audio as 0x03 frame; Dolphin expects 4ch audio from `SpeechSegment.raw_audio`
   - What's unclear: Dolphin's expected format — is it 16kHz S16LE interleaved or some other layout?
   - Recommendation: From prior phase research, Dolphin expects mono `[1, samples]` — the raw 4ch is for server-side processing, not fed directly to Dolphin as-is. No special format needed beyond "4ch S16LE interleaved".

4. **AudioTrack volume control**
   - What we know: The app has a TTS volume slider (0.0-1.0) stored in SharedPreferences as `tts_volume`
   - What's unclear: `AudioTrack` volume vs Android system volume — old code used `TextToSpeech.Engine.KEY_PARAM_VOLUME`
   - Recommendation: Use `audioTrack.setVolume(ttsVolume)` to apply the existing preference.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | JUnit 4 (Android) — `app/src/test/` for unit tests, `app/src/androidTest/` for instrumented |
| Config file | `app/build.gradle.kts` — `testImplementation(libs.junit)` already present |
| Quick run command | `cd smait-jackie-app && ./gradlew test` (unit tests, no device needed) |
| Full suite command | `./gradlew connectedAndroidTest` (requires device — LAB only) |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUD-01 | `CaeAudioManager` class compiles and can be instantiated (mocked context) | unit | `./gradlew test` | Wave 0 |
| AUD-02 | `adapt8chTo6chCaeFormat()` produces correct 24-byte frames with channel IDs 1..6 | unit | `./gradlew test` | Wave 0 |
| AUD-03 | `sendAudio(0x01)` and `sendRaw4ch(0x03)` produce frames with correct type bytes | unit | `./gradlew test` | Wave 0 |
| AUD-04 | DOA JSON frame has `{"type":"doa","angle":N,"beam":N}` structure | unit | `./gradlew test` | Wave 0 |
| TTS-04 | `handleBinaryMessage(0x05 + pcm)` routes PCM bytes to AudioTrack player | unit | `./gradlew test` | Wave 0 |

**Note:** Full hardware validation (CAE `onAudio` fires, AudioTrack plays speaker audio) is manual-only in LAB.

### Sampling Rate
- **Per task commit:** `./gradlew test` — unit tests for channel adapter, frame format, DOA JSON
- **Per wave merge:** `./gradlew test` (full unit suite)
- **Phase gate:** Unit tests green at home; hardware tests pass in lab before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `app/src/test/java/com/gow/smaitrobot/CaeAudioManagerTest.kt` — covers AUD-02, AUD-03, AUD-04
- [ ] `app/src/test/java/com/gow/smaitrobot/TtsAudioPlayerTest.kt` — covers TTS-04
- [ ] No framework install needed — JUnit 4 already in `build.gradle.kts`

---

## Key Facts for Planner

### What Already Exists (Do Not Rebuild)
- `app/libs/` — `cae.jar`, `AlsaRecorder.jar`, all `.so` files — PRESENT in main branch
- `app/src/main/assets/` — `hlw.ini`, `hlw.param`, `res_cae_model.bin`, `res_ivw_model.bin` — PRESENT in main branch
- `app/src/main/java/com/voice/` — `CaeCoreHelper.java`, `OnCaeOperatorlistener.java`, `FileUtil.java` — PRESENT in main branch
- `build.gradle.kts` — already has `cae.jar`, `AlsaRecorder.jar`, `jniLibs.srcDirs`, `abiFilters("armeabi-v7a")` — PRESENT
- `AndroidManifest.xml` — needs storage permissions check (may already be present from branch work)

### What Needs to Be Created/Modified
1. **`CaeAudioManager.kt`** — new file (adapt from branch, fix 3 bugs)
   - Fix: channel adapter format (add channel IDs)
   - Fix: DOA sent as JSON not binary
   - Add: raw 4ch stream on 0x03
2. **`TtsAudioPlayer.kt`** — new file (AudioTrack streaming for 0x05 frames)
3. **`MainActivity.kt`** — minimal edits:
   - Add `caeAudio: CaeAudioManager` field
   - Replace `startAudioCapture()` / `stopAudioCapture()` to use CaeAudioManager
   - Add `override fun onMessage(ws, bytes: ByteString)` to WebSocketListener
   - Wire `ttsAudioPlayer.stop()` on `tts_control: end` message

### Server-Side: No Changes Needed
The server already handles all streams correctly:
- 0x01 → `SPEECH_DETECTED` as `type="cae"` ✓
- 0x02 → `FACE_UPDATED` ✓
- 0x03 → `SPEECH_DETECTED` as `type="raw"` ✓
- JSON `{"type":"doa"}` → `DOA_UPDATE` event ✓
- 0x05 → sent by `ConnectionManager._on_tts_audio_chunk()` ✓

---

## Sources

### Primary (HIGH confidence)
- Codebase direct inspection — `smait-jackie-app/app/` (both branches)
- `docs/hardware-sdk/CAEDemoAIUI-4 MIC/CAEDemoAIUI/app/src/main/java/com/voice/caePk/CaeOperator.java` — reference implementation for channel adapter
- `smait/connection/protocol.py` — definitive frame type definitions
- `smait/connection/manager.py` — server DOA/binary frame handling
- `smait-jackie-app/CAE_INTEGRATION_GUIDE.md` — asset/file copy instructions
- `smait-jackie-app/CHANGES_TO_APPLY.md` — MainActivity wiring guide

### Secondary (MEDIUM confidence)
- Android AudioTrack API — standard Android SDK, well-documented behavior
- OkHttp WebSocketListener — `onMessage(ByteString)` override is documented API

### Tertiary (LOW confidence)
- CAE SDK `onAudio` output sample rate (24kHz vs 16kHz) — not verified from docs, needs lab measurement

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tech is already in the project, no new dependencies
- Architecture: HIGH — both branches inspected, server protocol fully documented
- Channel adapter fix: HIGH — reference implementation in CAE demo code confirmed
- DOA format fix: HIGH — server handler code directly confirmed JSON-only parsing
- AudioTrack setup: HIGH — standard Android API, 24kHz from Kokoro docs
- Auth token validity: LOW — cannot be verified without lab hardware
- CAE output sample rate: LOW — not confirmed from any source

**Research date:** 2026-03-10
**Valid until:** 2026-06-10 (stable proprietary SDK, unlikely to change)
