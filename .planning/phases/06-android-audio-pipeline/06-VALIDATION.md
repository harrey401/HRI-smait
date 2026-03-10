---
phase: 6
slug: android-audio-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 4 (Android) — `app/src/test/` for unit tests, `app/src/androidTest/` for instrumented |
| **Config file** | `app/build.gradle.kts` — `testImplementation(libs.junit)` already present |
| **Quick run command** | `cd smait-jackie-app && ./gradlew test` |
| **Full suite command** | `./gradlew connectedAndroidTest` (requires device — LAB only) |
| **Estimated runtime** | ~30 seconds (unit tests) |

---

## Sampling Rate

- **After every task commit:** Run `cd smait-jackie-app && ./gradlew test`
- **After every plan wave:** Run `cd smait-jackie-app && ./gradlew test` (full unit suite)
- **Before `/gsd:verify-work`:** Full suite must be green (unit at home, connectedAndroidTest in lab)
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | AUD-01 | unit | `./gradlew test` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | AUD-02 | unit | `./gradlew test` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | AUD-03 | unit | `./gradlew test` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | AUD-04 | unit | `./gradlew test` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | TTS-04 | unit | `./gradlew test` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `app/src/test/java/com/gow/smaitrobot/CaeAudioManagerTest.kt` — covers AUD-01, AUD-02, AUD-03, AUD-04
- [ ] `app/src/test/java/com/gow/smaitrobot/TtsAudioPlayerTest.kt` — covers TTS-04
- [ ] No framework install needed — JUnit 4 already in `build.gradle.kts`

*Existing infrastructure covers framework requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CAE `onAudio` fires with real microphone data | AUD-01 | Requires iFlytek RK3588 hardware | Connect robot, start app, verify logcat shows CAE callbacks |
| AudioTrack plays TTS audio from speaker | TTS-04 | Requires physical audio output | Send 0x05 frame from server, verify speaker output |
| Server demuxes all streams without errors | AUD-01..04 | Requires full robot+server setup | Connect app to server, verify all event types logged |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
