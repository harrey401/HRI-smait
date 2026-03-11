---
phase: 06-android-audio-pipeline
verified: 2026-03-10T21:30:00Z
status: gaps_found
score: 9/10 must-haves verified
gaps:
  - truth: "Unit tests verified to pass (compiled and executed)"
    status: partial
    reason: "Java/JDK not installed on home dev machine. Tests are substantively correct by code review but ./gradlew test and ./gradlew compileDebugKotlin cannot be executed in this environment. This is a LAB-gate constraint documented by both summaries."
    artifacts:
      - path: "smait-jackie-app/app/src/test/java/com/gow/smaitrobot/CaeAudioManagerTest.kt"
        issue: "Tests written and structurally correct; execution not verified (no JDK)"
      - path: "smait-jackie-app/app/src/test/java/com/gow/smaitrobot/TtsAudioPlayerTest.kt"
        issue: "Tests written and structurally correct; execution not verified (no JDK)"
    missing:
      - "Run ./gradlew test in lab with JDK installed to confirm all 18+11 unit tests pass"
      - "Run ./gradlew compileDebugKotlin to confirm project compiles without errors"
human_verification:
  - test: "Connect Android app to server, trigger TTS response"
    expected: "Audio plays on robot speaker at correct pitch (24kHz Kokoro output)"
    why_human: "Requires physical robot hardware with AudioTrack output; cannot verify PCM routing end-to-end programmatically"
  - test: "Make noise near mic array during active session, check DOA angle"
    expected: "Server receives JSON {type:doa, angle:N, beam:N} text frames and displays/logs DOA"
    why_human: "Requires physical 4-mic USB array; CAE onWakeup callback only fires with real hardware"
  - test: "Run ./gradlew test in lab with Android Studio (bundles JBR)"
    expected: "All 29 unit tests pass (18 in CaeAudioManagerTest + 11 in TtsAudioPlayerTest)"
    why_human: "JDK not available in home environment; confirmed blocker in both summaries"
---

# Phase 6: Android Audio Pipeline Verification Report

**Phase Goal:** Android app code updated for CAE beamforming, 3 streams, DOA, and AudioTrack TTS playback
**Verified:** 2026-03-10T21:30:00Z
**Status:** gaps_found (1 gap — environment constraint, not a code defect)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CaeAudioManager adapts 8ch ALSA to 6ch CAE with channel IDs 1..6 | VERIFIED | `adapt8chTo6chCaeFormat`: `out[dOff+1] = (ch+1).toByte()` — IDs 1..6 confirmed at line 88 |
| 2 | CAE beamformed output sent as 0x01 binary frame via WebSocket | VERIFIED | `onAudio` calls `sendAudio()` → `buildAudioFrame()` prepends `AUDIO_CAE_TYPE=0x01`; tested by `buildAudioFrame prepends 0x01 type byte` |
| 3 | Raw 4-channel audio extracted from 8ch ALSA and sent as 0x03 binary frame | VERIFIED | `pcmListener` calls `extract4chRaw(bytes)` then `sendRaw4ch()` → `buildRaw4chFrame()` prepends `AUDIO_RAW_TYPE=0x03` |
| 4 | DOA angles sent as JSON text frames with type=doa, not binary 0x03 | VERIFIED | `sendDoaAngle()` calls `webSocket?.send(buildDoaJson(...))` — String overload = text frame; no binary type byte; tested by `buildDoaJson does not start with binary type byte` |
| 5 | MainActivity uses CaeAudioManager instead of AudioRecord when connected | VERIFIED | `startAudioCapture()` creates `CaeAudioManager(this)` and calls `.start(webSocket!!)`; `AudioRecord` import absent; grep confirms only comment reference to AudioRecord |
| 6 | App receives binary 0x05 WebSocket frames and plays PCM16 audio via AudioTrack | VERIFIED | `onMessage(ws, bytes: ByteString)` override present; routes `0x05` frames to `ttsAudioPlayer?.handleBinaryFrame(data)` at lines 1069-1080 |
| 7 | AudioTrack configured for 24kHz mono PCM16 (matching Kokoro TTS output) | VERIFIED | `TtsAudioPlayer.start()`: `setSampleRate(24000)`, `ENCODING_PCM_16BIT`, `CHANNEL_OUT_MONO`, `MODE_STREAM` |
| 8 | AudioTrack.write() runs on background HandlerThread | VERIFIED | `write()` posts `track.write(slice, 0, slice.size)` to `playbackHandler`; handler backed by `HandlerThread("TtsPlayback")` |
| 9 | TTS playback respects ttsVolume preference (0.0-1.0) | VERIFIED | `ttsAudioPlayer?.setVolume(ttsVolume)` in `onCreate`; volume slider `onProgressChanged` calls `ttsAudioPlayer?.setVolume(ttsVolume)` at line 355 |
| 10 | Playback stops cleanly on tts_control:end | VERIFIED | `handleServerMessage` dispatches `tts_control` → `action=end` → `ttsAudioPlayer?.stop(); isSpeaking.set(false)` at lines 1170-1176 |

**Score:** 9/10 truths verified as code-correct; 1 truth (unit test execution) pending lab validation due to JDK absence

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `smait-jackie-app/app/src/main/java/com/gow/smaitrobot/CaeAudioManager.kt` | 150 | 375 | VERIFIED | Full implementation: channel adapter, dual stream, JSON DOA, lifecycle management |
| `smait-jackie-app/app/src/test/java/com/gow/smaitrobot/CaeAudioManagerTest.kt` | 80 | 215 | VERIFIED (unexecuted) | 18 JUnit 4 tests; structurally correct; cannot run without JDK |
| `smait-jackie-app/app/src/main/java/com/gow/smaitrobot/TtsAudioPlayer.kt` | 60 | 204 | VERIFIED | Full implementation: injectable audioWriter, AudioTrack 24kHz PCM16, HandlerThread |
| `smait-jackie-app/app/src/test/java/com/gow/smaitrobot/TtsAudioPlayerTest.kt` | 40 | 125 | VERIFIED (unexecuted) | 11 JUnit 4 tests; structurally correct; cannot run without JDK |
| `smait-jackie-app/app/src/main/java/com/gow/smaitrobot/MainActivity.kt` | - | ~1100 | VERIFIED | CaeAudioManager wired; TtsAudioPlayer wired; AudioRecord fully removed |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| CaeAudioManager.kt | CaeCoreHelper.java | `writeAudio(adapted)` in pcmListener | WIRED | Line 298: `caeCoreHelper?.writeAudio(adapted)` |
| CaeAudioManager.kt | WebSocket | `sendAudio` (0x01), `sendRaw4ch` (0x03), `sendDoaAngle` (text) | WIRED | Lines 312, 325, 340: all three send paths confirmed with correct frame types |
| MainActivity.kt | CaeAudioManager.kt | `caeAudioManager.start(ws)` replaces `startAudioCapture()` | WIRED | Lines 1011-1012: `caeAudioManager = CaeAudioManager(this); caeAudioManager!!.start(webSocket!!)` |
| MainActivity.kt | TtsAudioPlayer.kt | `onMessage(ws, bytes)` dispatches 0x05 to `ttsAudioPlayer.handleBinaryFrame()` | WIRED | Lines 1069-1079: `when(data[0]) { 0x05.toByte() -> ttsAudioPlayer?.handleBinaryFrame(data) }` |
| MainActivity.kt | TtsAudioPlayer.kt | `tts_control:end` calls `ttsAudioPlayer.stop()` | WIRED | Lines 1170-1176: `ttsAudioPlayer?.stop(); isSpeaking.set(false)` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| AUD-01 | 06-01-PLAN.md | Android app integrates CAE beamforming | SATISFIED | CaeAudioManager replaces AudioRecord; CAE bypass removed; `onAudio` sends real beamformed output |
| AUD-02 | 06-01-PLAN.md | CAE 8-channel to 4-channel format mismatch resolved | SATISFIED | `adapt8chTo6chCaeFormat` uses `[0x00, ch_id, pcm_lo, pcm_hi]` with IDs 1..6; critical fix from branch applied |
| AUD-03 | 06-01-PLAN.md | App sends 3 streams: CAE audio (0x01), raw 4-channel (0x03), video (0x02) | SATISFIED | `pcmListener` dual-path: adapt→writeAudio AND extract4chRaw→sendRaw4ch(0x03); onAudio sends 0x01; video (0x02) unchanged |
| AUD-04 | 06-01-PLAN.md | App sends DOA angles from CAE callbacks as JSON messages | SATISFIED | `onWakeup` → `sendDoaAngle()` → `webSocket?.send(buildDoaJson(...))` — text frame, not binary |
| TTS-04 | 06-02-PLAN.md | Android app plays TTS audio via AudioTrack on speaker | SATISFIED | TtsAudioPlayer with AudioTrack 24kHz PCM16 MODE_STREAM; binary 0x05 handler in MainActivity |

**No orphaned requirements:** All 5 Phase 6 requirements (AUD-01, AUD-02, AUD-03, AUD-04, TTS-04) are claimed by plans and have implementation evidence. REQUIREMENTS.md traceability table confirms all 5 assigned to Phase 6.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | - | - | - | - |

No TODOs, FIXMEs, placeholders, empty implementations, or console-log-only stubs detected in any of the four created/modified files. The old TextToSpeech code retained in MainActivity is intentional (plan specifies removal deferred to Phase 8 QUAL-02); not an anti-pattern.

### Human Verification Required

#### 1. Unit Test Execution (Lab Gate)

**Test:** In lab environment with Android Studio (which bundles JBR), run `./gradlew test`
**Expected:** All 29 unit tests pass — 18 in CaeAudioManagerTest (channel adapter, extract4ch, frame type bytes, DOA JSON) and 11 in TtsAudioPlayerTest (frame routing, volume clamping)
**Why human:** JDK not installed on home dev machine. Both summaries document this as a known environment constraint with lab validation required. Code correctness verified by static analysis.

#### 2. CAE Beamforming Hardware Test

**Test:** With 4-mic USB array connected, run app and speak near the mic. Check server logs for audio frames.
**Expected:** Server receives continuous 0x01 binary frames (CAE beamformed audio) AND 0x03 binary frames (raw 4ch). DOA angles arrive as JSON text frames `{"type":"doa","angle":N,"beam":N}`.
**Why human:** CAE `onAudio` and `onWakeup` callbacks only fire with real iFlytek hardware. Cannot simulate ALSA device or CAE engine in software.

#### 3. TTS AudioTrack Playback

**Test:** Trigger a TTS response from the server. Observe robot speaker output.
**Expected:** Audio plays on robot speaker at correct pitch. Stopping mid-sentence via barge-in or `tts_control:end` message cleanly halts playback.
**Why human:** AudioTrack output requires physical speaker and Android hardware. Cannot verify PCM playback quality, volume levels, or background thread behavior without device.

### Gaps Summary

One gap exists: **unit test execution is unverified** because JDK is not installed in the home development environment. This is explicitly a LAB-gate constraint documented in both plan summaries and consistent with the Phase 6 architecture (HOME=code, LAB=hardware validation). All test logic is structurally sound — `CaeAudioManagerTest` correctly calls companion object functions (`adapt8chTo6chCaeFormat`, `extract4chRaw`, `buildAudioFrame`, `buildRaw4chFrame`, `buildDoaJson`) and verifies expected byte-level behavior; `TtsAudioPlayerTest` uses the injectable `audioWriter` lambda design to test frame routing without Android SDK dependencies.

All code-verifiable must-haves are satisfied: artifacts exist, are substantive (well above minimum line counts), and are correctly wired. The four committed hashes (9c11dac, 7955d03, d28d5fb, bd49250) all exist in the smait-jackie-app git history.

---

_Verified: 2026-03-10T21:30:00Z_
_Verifier: Claude (gsd-verifier)_
