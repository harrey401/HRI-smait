---
phase: 12
slug: android-app-rebuild-and-wie-theme-home
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 4 + Robolectric 4.13 + Turbine 1.2 + Compose Test |
| **Config file** | `app/src/test/` (JVM unit tests) + `app/src/androidTest/` (instrumented) |
| **Quick run command** | `./gradlew :app:testDebugUnitTest` |
| **Full suite command** | `./gradlew :app:test :app:connectedDebugAndroidTest` |
| **Estimated runtime** | ~15 seconds (JVM), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `./gradlew :app:testDebugUnitTest`
- **After every plan wave:** Run `./gradlew :app:test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | APP-01 | build | `./gradlew :app:assembleDebug` | Wave 0 | ⬜ pending |
| 12-01-02 | 01 | 1 | APP-02 | unit | `./gradlew test --tests "ThemeRepositoryTest"` | Wave 0 | ⬜ pending |
| 12-01-03 | 01 | 1 | APP-09 | unit | `./gradlew test --tests "ThemeRepositoryTest"` | Wave 0 | ⬜ pending |
| 12-01-04 | 01 | 1 | WIE-01 | unit | `./gradlew test --tests "WiEThemeTest"` | Wave 0 | ⬜ pending |
| 12-01-05 | 01 | 1 | WIE-02 | unit | `./gradlew test --tests "WiEThemeTest"` | Wave 0 | ⬜ pending |
| 12-02-01 | 02 | 1 | APP-08 | unit | `./gradlew test --tests "WebSocketRepositoryTest"` | Wave 0 | ⬜ pending |
| 12-02-02 | 02 | 1 | APP-08 | unit | `./gradlew test --tests "TtsAudioPlayerTest"` | ✅ EXISTS | ⬜ pending |
| 12-02-03 | 02 | 1 | APP-08 | unit | `./gradlew test --tests "CaeAudioManagerTest"` | ✅ EXISTS | ⬜ pending |
| 12-03-01 | 03 | 2 | APP-03 | unit | `./gradlew test --tests "HomeScreenTest"` | Wave 0 | ⬜ pending |
| 12-03-02 | 03 | 2 | APP-05 | unit | `./gradlew test --tests "ConversationViewModelTest"` | Wave 0 | ⬜ pending |
| 12-03-03 | 03 | 2 | APP-04 | unit | `./gradlew test --tests "NavigationMapViewModelTest"` | Wave 0 | ⬜ pending |
| 12-03-04 | 03 | 2 | APP-06 | unit | `./gradlew test --tests "FacilitiesViewModelTest"` | Wave 0 | ⬜ pending |
| 12-03-05 | 03 | 2 | APP-07 | unit | `./gradlew test --tests "EventInfoViewModelTest"` | Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `app/src/test/java/com/gow/smaitrobot/ThemeRepositoryTest.kt` — stubs for APP-02, APP-09, WIE-01, WIE-02
- [ ] `app/src/test/java/com/gow/smaitrobot/WebSocketRepositoryTest.kt` — stubs for APP-08 (new WS layer)
- [ ] `app/src/test/java/com/gow/smaitrobot/HomeScreenTest.kt` — stubs for APP-03 (Robolectric)
- [ ] `app/src/test/java/com/gow/smaitrobot/ConversationViewModelTest.kt` — stubs for APP-05
- [ ] `app/src/test/java/com/gow/smaitrobot/NavigationMapViewModelTest.kt` — stubs for APP-04
- [ ] `app/src/test/java/com/gow/smaitrobot/FacilitiesViewModelTest.kt` — stubs for APP-06
- [ ] `app/src/test/java/com/gow/smaitrobot/EventInfoViewModelTest.kt` — stubs for APP-07
- [ ] Robolectric setup: `testImplementation("org.robolectric:robolectric:4.13")` + config
- [ ] Turbine setup: `testImplementation("app.cash.turbine:turbine:1.2.0")`

*Existing tests `TtsAudioPlayerTest.kt` and `CaeAudioManagerTest.kt` ALREADY EXIST — will continue passing.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| App renders correctly in landscape on Jackie's screen | APP-01 | Device-specific layout | Install APK on Jackie, verify landscape immersive mode |
| Selfie capture saves photo | APP-05 | Requires camera hardware | Open conversation → tap camera icon → verify countdown + capture |
| Sponsor bar scrolls smoothly | APP-03 | Visual UX quality | Add 10+ sponsors, verify slow horizontal scroll on device |
| Lottie robot face animation transitions | APP-05 | Visual UX quality | Trigger listening/thinking/speaking states, verify smooth transitions |
| Theme swap at runtime | APP-02, APP-09 | Visual verification | Replace wie2026_theme.json with default, verify colors change |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
