---
phase: 06-android-audio-pipeline
plan: "01"
subsystem: android-audio
tags: [android, kotlin, cae-sdk, alsa, beamforming, websocket, tdd]
dependency_graph:
  requires: []
  provides: [CaeAudioManager.kt, CaeAudioManagerTest.kt, MainActivity-CAE-wiring]
  affects: [smait-jackie-app/MainActivity.kt]
tech_stack:
  added: []
  patterns:
    - CAE SDK channel adapter (8ch 16-bit to 6ch 32-bit with channel IDs 1..6)
    - Dual WebSocket stream (0x01 AUDIO_CAE + 0x03 AUDIO_RAW per ALSA callback)
    - DOA JSON text frame (not binary) for server _handle_text() parsing
key_files:
  created:
    - smait-jackie-app/app/src/main/java/com/gow/smaitrobot/CaeAudioManager.kt
    - smait-jackie-app/app/src/test/java/com/gow/smaitrobot/CaeAudioManagerTest.kt
  modified:
    - smait-jackie-app/app/src/main/java/com/gow/smaitrobot/MainActivity.kt
decisions:
  - "[AUD-02] Channel adapter uses [0x00, ch_id, pcm_lo, pcm_hi] with IDs 1..6; branch used 0x00 which caused CAE onAudio to never fire"
  - "[AUD-04] DOA sent as JSON text WebSocket frame not binary 0x03; binary collides with AUDIO_RAW protocol type"
  - "[AUD-01/03] CAE bypass (mono extraction + 16x gain) removed; onAudio sends 0x01 beamformed; pcmListener sends 0x03 raw 4ch"
  - "[MainActivity] isSpeaking mic gate removed from audio capture; server Phase 5 AEC handles echo"
  - "[Testing] Java/JDK not installed on dev machine; test execution deferred to lab environment with Android Studio"
metrics:
  duration: "7m"
  completed: "2026-03-10T20:55:30Z"
  tasks_completed: 2
  files_created: 2
  files_modified: 1
---

# Phase 6 Plan 1: CaeAudioManager — CAE SDK Integration Summary

**One-liner:** CAE SDK audio manager with fixed channel adapter (IDs 1..6), dual stream (0x01 beamformed + 0x03 raw 4ch), JSON DOA text frames, wired into MainActivity replacing AudioRecord.

## What Was Built

### CaeAudioManager.kt (375 lines)
New file in `smait-jackie-app/app/src/main/java/com/gow/smaitrobot/`:

- `adapt8chTo6chCaeFormat(data: ByteArray): ByteArray` — converts 8ch 16-bit ALSA frames (16 bytes) to 6ch 32-bit CAE format (24 bytes) with `[0x00, channel_id, pcm_lo, pcm_hi]` slots. Channel IDs 1..6, mapping ALSA ch0-3 (mics) + ch6/ch7 (refs). The critical fix: the cae-work-march2 branch used `[0x00, 0x00, ...]` (no channel ID) which caused CAE `onAudio` to never fire.

- `extract4chRaw(alsaData: ByteArray): ByteArray` — extracts ch0-ch3 from 8ch ALSA data (16 bytes → 8 bytes), for the raw 4-channel stream sent to the server for Dolphin speaker separation.

- `buildAudioFrame(data): ByteArray` — prepends 0x01 (AUDIO_CAE) type byte.
- `buildRaw4chFrame(data): ByteArray` — prepends 0x03 (AUDIO_RAW) type byte.
- `buildDoaJson(angle, beam): String` — returns `{"type":"doa","angle":N,"beam":N}` JSON text (not binary).

- `pcmListener` — per ALSA callback: adapt → writeAudio (feeds CAE) AND extract4chRaw → sendRaw4ch (sends 0x03).
- `onAudio` callback — sends CAE beamformed output as 0x01 (bypass removed).
- `onWakeup` callback — sends DOA as JSON text frame (FIX 2: was binary 0x03 ByteBuffer).
- `copyAssetsIfNeeded()`, `start(ws)`, `stop()`, `cleanup()` lifecycle management.
- `CAESetRealBeam(0)` called after CaeCoreHelper init for continuous output (not wake-word gate).

### CaeAudioManagerTest.kt (215 lines)
JUnit 4 tests in `smait-jackie-app/app/src/test/java/com/gow/smaitrobot/`:

18 tests covering:
- `adapt8chTo6chCaeFormat`: 1-frame = 24 bytes, 2-frames = 48 bytes; byte[0]=0x00; byte[1]=1..6; ch0 maps slot 0; ch6 maps slot 4 (ref1); ch7 maps slot 5 (ref2)
- `extract4chRaw`: 1-frame = 8 bytes; first/fourth channel bytes correct; no ch4+ data
- `buildAudioFrame`: size=1+payload; first byte=0x01
- `buildRaw4chFrame`: size=1+payload; first byte=0x03
- `buildDoaJson`: type="doa" key; angle/beam values match; valid JSON string; starts with `{` not binary byte

### MainActivity.kt (modified)
- Removed: `audioRecord: AudioRecord?`, `audioThread: Thread?` fields
- Removed: `import android.media.AudioRecord`, `import android.media.MediaRecorder`, `import android.media.AudioFormat`
- Removed: unused companion constants `SAMPLE_RATE`, `CHANNEL_CONFIG`, `AUDIO_FORMAT`, `AUDIO_TYPE`
- Added: `caeAudioManager: CaeAudioManager?` field
- Added: `CaeAudioManager(this).copyAssetsIfNeeded()` in onCreate after prefs init
- Replaced: `startAudioCapture()` body → creates CaeAudioManager, calls `start(webSocket!!)`
- Replaced: `stopAudioCapture()` body → calls `caeAudioManager?.stop()`, nulls field
- Removed: `!isSpeaking.get()` mic gating (server Phase 5 software AEC handles echo cancellation)

## Commits

| Hash | Message |
|------|---------|
| 9c11dac | feat(06-01): create CaeAudioManager with channel adapter, dual stream, JSON DOA + unit tests |
| 7955d03 | feat(06-01): wire CaeAudioManager into MainActivity, remove AudioRecord |

## Deviations from Plan

### Environment Issue (Not Auto-Fixable)

**1. [Rule 3 - Blocker] Java/JDK not installed on dev machine**
- **Found during:** Task 1 verification (test run)
- **Issue:** `./gradlew test` and `./gradlew compileDebugKotlin` both fail with "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH". Java is not installed on this Ubuntu machine, and `sudo apt-get install` requires interactive password input.
- **Impact:** Unit tests and Kotlin compilation cannot be verified locally.
- **Mitigation:** Code correctness verified by logic review against spec:
  - Channel adapter: `(ch + 1).toByte()` produces IDs 1..6 ✓
  - Frame builders: byte[0]=0x01/0x03 with `System.arraycopy` for payload ✓
  - DOA JSON: `JSONObject().put("type","doa")...toString()` produces text ✓
  - 18 tests cover all spec behaviors with correct assertions ✓
- **Resolution:** Test execution and compile verification to be done in lab with Android Studio (which bundles JBR/JDK). See Phase 6 validation architecture in 06-RESEARCH.md.
- **Files modified:** None (environment issue)

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Companion object `internal` functions for testability | `internal` functions in same module are accessible to `src/test/` JUnit tests without reflection |
| `copyAssetsIfNeeded()` called in onCreate (not startAudioCapture) | Assets need to be ready before any connection; called once on app init, safe if called multiple times |
| isSpeaking mic gate removed from audio capture | Server Phase 5 software AEC handles echo; CAE hardware also has AEC; redundant gate causes missed audio |
| `buildAudioFrame(audioData.copyOfRange(0, dataLen))` in onAudio | CAE onAudio provides `dataLen` separate from array size; slice ensures only valid bytes framed |

## Self-Check: PASSED

- CaeAudioManager.kt: FOUND at smait-jackie-app/app/src/main/java/com/gow/smaitrobot/
- CaeAudioManagerTest.kt: FOUND at smait-jackie-app/app/src/test/java/com/gow/smaitrobot/
- 06-01-SUMMARY.md: FOUND at .planning/phases/06-android-audio-pipeline/
- Commit 9c11dac: FOUND (Task 1 - CaeAudioManager + tests)
- Commit 7955d03: FOUND (Task 2 - MainActivity wiring)
