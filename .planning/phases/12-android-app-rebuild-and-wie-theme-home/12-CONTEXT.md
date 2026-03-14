# Phase 12: Android App Rebuild and WiE Theme - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Full rewrite of Jackie's Android touchscreen app from XML/imperative Kotlin to Jetpack Compose + MVVM. Delivers 5 screens (Home, Conversation, Navigation, Facilities, Event Info) with WiE 2026 branding, event-configurable theming via JSON, and all existing WebSocket streams preserved. HOME phase — no robot hardware needed.

</domain>

<decisions>
## Implementation Decisions

### Screen Flow & Navigation
- Bottom navigation bar with 5 destinations: Home, Chat, Map, Facilities, Event Info
- Nav bar always visible on all screens (never hides)
- Home screen IS the idle screen — no separate idle/welcome view
- Auto-return to Home screen on session end OR after ~30s silence timeout (clean slate for next person)
- Home screen cards are a mix: some navigate to bottom nav destinations (e.g., "Ask Me Anything" → Chat), some open inline content (e.g., "Keynote Info" opens event content)

### Home Screen Layout
- SJSU + BioRob lab logos at top
- 2x3 or 2x4 grid of square tiles (event-specific cards)
- Sponsor bar at bottom: static row of logos, slow horizontal scroll if too many to fit
- Sponsor bar + logos appear on Home and Event Info screens only — not on Chat, Map, or Facilities

### Visual Identity & Theming
- Clean Material 3 design language
- Light mode default
- WiE 2026 palette: purple/teal/orange with "Engineering Beyond Imagination" branding
- Equal branding split: WiE branding prominent on Home + Event Info, robot personality takes over on Chat/Map/Facilities
- Theme JSON config controls: colors, logo, event name, card labels/text — NOT layout or card definitions
- Swapping the JSON file changes the event branding with no code changes

### Conversation Screen
- Chat bubble style: user messages in one color, robot messages in another, scrolling up
- Animated robot face displayed on one side of the screen — reacts to robot state (listening, thinking, speaking)
- Lip sync on the animated face is a stretch goal (sync mouth to TTS audio if feasible)
- No status indicator in the app UI — Jackie's physical LEDs (ears/eyes) indicate listening/thinking/speaking states
- Camera preview hidden by default, toggle-able via a small button (useful for demos)

### Kiosk UX & Accessibility
- Large touch targets: 60-80px minimum tiles with 5mm+ padding between tappable elements
- Visual tap feedback: ripple/press animation on every touchable element for immediate confirmation
- Accessible text sizing: 18sp minimum for body text, 24sp+ for labels and headings
- High contrast for varying conference room lighting conditions

### Selfie Feature
- Camera icon on conversation screen
- 3-2-1 countdown with flash animation
- Preview with retake/save buttons
- Saves to device storage
- Same proven flow as current app

### Post-Session Feedback
- Triggered when conversation session ends (before auto-return to Home)
- Step 1: 1-5 star rating (one tap)
- Step 2: "Want to answer 2-3 quick questions?" button — optional, skip returns to Home
- Survey questions hardcoded (2-3 short research questions)
- Star rating auto-dismisses after ~10 seconds if no interaction
- Feedback data sent to SMAIT server via new WebSocket JSON message type
- Local storage fallback if WebSocket disconnected — sync when reconnected

### WebSocket Protocol Preservation
- All existing binary streams must work identically: 0x01 (CAE audio), 0x02 (video), 0x03 (raw 4ch), 0x05 (TTS), 0x06 (map PNG)
- All existing JSON text messages preserved: state, transcript, response, tts_control, nav_status, doa, cae_status
- New JSON message type added: feedback (rating + survey responses)
- OkHttp3 WebSocket client with auto-reconnect (existing pattern)
- CaeAudioManager.kt and TtsAudioPlayer.kt reusable as-is — wrap into MVVM architecture

### Claude's Discretion
- Exact Material 3 component choices (cards, surfaces, typography variants)
- Compose navigation library implementation details
- MVVM layer structure (ViewModel scoping, repository pattern)
- Animated robot face implementation approach (Lottie, Canvas, custom Compose animation)
- Lip sync feasibility assessment and fallback if too complex
- Chat bubble exact styling and spacing
- Sponsor bar scroll animation speed
- Survey question wording
- Feedback WebSocket message schema
- How to structure the theme JSON file internally

</decisions>

<specifics>
## Specific Ideas

- The idle screen and home screen are the same thing — Jackie always shows the interactive grid, no "approach me" barrier
- Sponsor bar: static row when sponsors fit, slow auto-scroll when they overflow — not fast or distracting
- Status indication (listening/thinking/speaking) is done via Jackie's physical ear and eye LEDs, NOT in the app UI
- Selfie feature is a crowd-pleaser at events — keep it accessible from the conversation screen
- Feedback collection is for research data — star rating is quick, survey is opt-in so it doesn't annoy casual users
- Current app's CaeAudioManager and TtsAudioPlayer are clean Kotlin classes that can be wrapped into the new architecture without rewriting

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CaeAudioManager.kt` (376 lines): CAE SDK audio pipeline — ALSA recording, beamforming, DOA. Reuse as-is, wrap in ViewModel
- `TtsAudioPlayer.kt` (225 lines): AudioTrack PCM16 streaming for 0x05 frames. Reuse as-is, wrap in ViewModel
- WebSocket client pattern (OkHttp3 with auto-reconnect, SharedPreferences for IP/port config)
- Camera2 API setup for video capture (YUV_420_888 → JPEG → 0x02 frames)
- Chat message data model and adapter logic (can inform Compose list item design)

### Established Patterns
- Single Activity architecture (keep this — Compose navigation within one activity)
- Binary WebSocket frame protocol with type byte prefix
- JSON text messages for state/transcript/response/control
- SharedPreferences for server IP/port configuration
- Runtime permission handling (CAMERA, RECORD_AUDIO, INTERNET)

### Integration Points
- WebSocket connection: app connects to SMAIT server at configurable IP:port
- Audio input: CaeAudioManager sends 0x01 + 0x03 frames to server
- Audio output: TtsAudioPlayer receives 0x05 frames from server
- Video: Camera2 sends 0x02 JPEG frames to server
- Map display: receives 0x06 PNG from server for navigation screen
- JSON messages: receives state/transcript/response/nav_status from server; sends doa/cae_status/feedback to server
- Local JARs: cae.jar and AlsaRecorder.jar in libs/ (CAE SDK + ALSA recording)

</code_context>

<deferred>
## Deferred Ideas

- QR code on conversation screen for phone handoff (continue conversation/get directions on phone)
- Multilingual UI hint ("Speak in any language" label)
- Session summary card with QR code after conversation ends
- Navigation progress pill on home screen ("Heading to Room 204...")
- Multi-language UI labels (not just speech)
- Guest analytics dashboard (conversation count, popular questions, peak times)
- Patrol/attract mode (robot moves around, attract animation on screen)

</deferred>

---

*Phase: 12-android-app-rebuild-and-wie-theme-home*
*Context gathered: 2026-03-14*
