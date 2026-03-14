# Phase 12: Android App Rebuild and WiE Theme - Research

**Researched:** 2026-03-14
**Domain:** Android / Jetpack Compose / MVVM / WebSocket / Material 3
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Screen Flow & Navigation**
- Bottom navigation bar with 5 destinations: Home, Chat, Map, Facilities, Event Info
- Nav bar always visible on all screens (never hides)
- Home screen IS the idle screen — no separate idle/welcome view
- Auto-return to Home screen on session end OR after ~30s silence timeout (clean slate for next person)
- Home screen cards are a mix: some navigate to bottom nav destinations (e.g., "Ask Me Anything" → Chat), some open inline content (e.g., "Keynote Info" opens event content)

**Home Screen Layout**
- SJSU + BioRob lab logos at top
- 2x3 or 2x4 grid of square tiles (event-specific cards)
- Sponsor bar at bottom: static row of logos, slow horizontal scroll if too many to fit
- Sponsor bar + logos appear on Home and Event Info screens only

**Visual Identity & Theming**
- Clean Material 3 design language
- Light mode default
- WiE 2026 palette: purple/teal/orange with "Engineering Beyond Imagination" branding
- Equal branding split: WiE branding on Home + Event Info, robot personality on Chat/Map/Facilities
- Theme JSON config controls: colors, logo, event name, card labels/text — NOT layout or card definitions
- Swapping the JSON file changes event branding with no code changes

**Conversation Screen**
- Chat bubble style: user messages one color, robot messages another, scrolling up
- Animated robot face displayed on one side — reacts to robot state (listening, thinking, speaking)
- Lip sync on animated face is a STRETCH GOAL
- No status indicator in app UI — Jackie's physical LEDs handle that
- Camera preview hidden by default, toggle-able via small button

**Kiosk UX & Accessibility**
- Large touch targets: 60-80px minimum tiles with 5mm+ padding between tappable elements
- Visual tap feedback: ripple/press animation on every touchable element
- Accessible text sizing: 18sp minimum body text, 24sp+ for labels/headings
- High contrast for varying conference room lighting

**Selfie Feature**
- Camera icon on conversation screen
- 3-2-1 countdown with flash animation
- Preview with retake/save buttons
- Saves to device storage
- Same proven flow as current app

**Post-Session Feedback**
- Triggered when conversation session ends (before auto-return to Home)
- Step 1: 1-5 star rating (one tap)
- Step 2: optional 2-3 question survey
- Star rating auto-dismisses after ~10 seconds if no interaction
- Feedback sent via new WebSocket JSON message type: `feedback`
- Local storage fallback if disconnected — sync when reconnected

**WebSocket Protocol Preservation**
- All existing binary streams identical: 0x01 (CAE audio), 0x02 (video), 0x03 (raw 4ch), 0x05 (TTS), 0x06 (map PNG)
- All existing JSON text messages preserved: state, transcript, response, tts_control, nav_status, doa, cae_status
- New JSON message type: `feedback` (rating + survey responses)
- OkHttp3 WebSocket client with auto-reconnect
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

### Deferred Ideas (OUT OF SCOPE)
- QR code on conversation screen for phone handoff
- Multilingual UI hint
- Session summary card with QR code after conversation ends
- Navigation progress pill on home screen
- Multi-language UI labels
- Guest analytics dashboard
- Patrol/attract mode
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| APP-01 | Complete app rewrite with Jetpack Compose, MVVM | Compose BOM 2026.03.00 + Navigation 2.8+, MVVM with StateFlow |
| APP-02 | Base design system: colors, fonts, logos swappable per event | Material 3 MaterialTheme + JSON asset parsing |
| APP-03 | Home screen with event-customizable cards | LazyVerticalGrid + JSON-driven card config |
| APP-04 | Navigation screen: live map, robot position, path, destination | AndroidView(ImageView) or Compose Image with bitmap state |
| APP-05 | Conversation UI: transcript, mic indicator, robot avatar | LazyColumn + Lottie avatar + StateFlow transcript |
| APP-06 | Facilities/wayfinding screen: searchable POI list + "take me there" | LazyColumn + filter state + WebSocket navigate_to trigger |
| APP-07 | Event info screen: schedule, speakers, venue map from config | JSON-driven content, Image composable for venue map |
| APP-08 | All existing audio/video/TTS WebSocket streams preserved | WebSocketRepository wraps OkHttp3, CaeAudioManager+TtsAudioPlayer reused |
| APP-09 | Event theming via JSON config file | ThemeRepository reads assets/theme.json at startup |
| WIE-01 | WiE 2026 theme: purple/teal/orange, "Engineering Beyond Imagination" | assets/wie2026_theme.json with Material 3 color tokens |
| WIE-02 | WiE-specific cards: session tracks, career panels, keynote info, venue directions | wie2026_theme.json cards[] array drives Home screen grid |
</phase_requirements>

---

## Summary

Phase 12 is a full Android app rewrite targeting Jetpack Compose + MVVM, replacing a 1376-line monolithic `MainActivity.kt` with a proper multi-screen architecture. The app has two non-negotiable constraints: (1) it must preserve the binary WebSocket protocol exactly (0x01/0x02/0x03/0x05/0x06 frame types and all JSON message types), and (2) it must load all event-specific UI content (colors, logos, card text, schedule) from a JSON config file with zero code changes between events.

The existing `CaeAudioManager.kt` and `TtsAudioPlayer.kt` are well-isolated Kotlin classes with pure-function APIs and existing unit tests — they can be moved directly into the new project and wrapped in ViewModels. The WebSocket client (OkHttp3) will be extracted into a repository layer. Navigation uses `androidx.navigation:navigation-compose` with type-safe routes (sealed class destinations). Theming uses Material 3 `MaterialTheme` with a `ColorScheme` and `Typography` built at runtime from the JSON config.

The animated robot face is best implemented with Lottie (`com.airbnb.android:lottie-compose`): predefine 3-4 animation states (idle, listening, thinking, speaking), and drive transitions from a `robotState` StateFlow in the ViewModel. Lip sync is a stretch goal — if attempted, use a phoneme-to-animation mapping triggered from TTS audio amplitude; otherwise the 4-state face is sufficient.

**Primary recommendation:** Single-activity Compose app, 5-tab BottomNavigation, one `AppViewModel` per screen (scoped to NavBackStackEntry), `WebSocketRepository` as singleton in Application-scope, `ThemeRepository` loads JSON from `assets/` on startup and exposes `AppThemeConfig` as `StateFlow`. Build with Compose BOM 2026.03.00.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Compose BOM | 2026.03.00 | Version management for all Compose artifacts | Single version declaration, Google-blessed compatibility |
| androidx.compose.material3 | 1.4.0 (via BOM) | Material 3 UI components + theming | Current standard for new Android apps |
| androidx.navigation:navigation-compose | 2.8.x | Type-safe Compose navigation with NavHost | Official Jetpack navigation for Compose |
| kotlinx-serialization | 1.7.x | @Serializable type-safe nav routes | Required for Navigation 2.8 type-safe API |
| com.airbnb.android:lottie-compose | 6.6.x | Animated robot avatar | Lightweight vector animations, Compose-native API |
| com.squareup.okhttp3:okhttp | 4.12.0 | WebSocket client (PRESERVE — already working) | Proven in existing app, no reason to change |
| org.jetbrains.kotlinx:kotlinx-coroutines-android | 1.8.x | ViewModel + Flow + WebSocket threading | Standard coroutines for Android |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| androidx.lifecycle:lifecycle-viewmodel-compose | 2.8.x | viewModel() composable + rememberSaveable integration | Every screen ViewModel |
| androidx.lifecycle:lifecycle-runtime-compose | 2.8.x | collectAsStateWithLifecycle() | Lifecycle-safe StateFlow collection |
| com.google.code.gson:gson OR org.json | 2.10.x / built-in | JSON theme config parsing | Gson for type-safe data classes; org.json already on device (no dep needed) |
| app.cash.turbine:turbine | 1.2.x | StateFlow testing | testOnly — makes Flow emission assertions concise |
| org.robolectric:robolectric | 4.13.x | JVM Compose UI testing without emulator | Unit-test composables without device |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| lottie-compose | Compose Canvas custom draw | Canvas is more control, but requires animator logic from scratch; Lottie has robot/avatar assets free on LottieFiles |
| OkHttp3 WebSocket | Ktor WebSocket | OkHttp3 already proven in existing app with exact binary protocol; no migration risk |
| org.json (built-in) | Gson / Moshi | org.json needs no dependency, but no data class deserialization; Gson adds 100KB but cleaner theme config parsing — recommend Gson |

**Installation (build.gradle.kts):**
```kotlin
// Version catalog: gradle/libs.versions.toml
[versions]
compose-bom = "2026.03.00"
navigation-compose = "2.8.9"   # verify latest stable at time of build
lottie-compose = "6.6.0"
okhttp = "4.12.0"
coroutines = "1.8.1"
kotlin-serialization = "1.7.3"
gson = "2.11.0"

[libraries]
compose-bom = { group = "androidx.compose", name = "compose-bom", version.ref = "compose-bom" }
compose-material3 = { group = "androidx.compose.material3", name = "material3" }
compose-ui = { group = "androidx.compose.ui", name = "ui" }
navigation-compose = { group = "androidx.navigation", name = "navigation-compose", version.ref = "navigation-compose" }
lottie-compose = { group = "com.airbnb.android", name = "lottie-compose", version.ref = "lottie-compose" }
okhttp = { group = "com.squareup.okhttp3", name = "okhttp", version.ref = "okhttp" }
gson = { group = "com.google.code.gson", name = "gson", version.ref = "gson" }
kotlinx-serialization-json = { group = "org.jetbrains.kotlinx", name = "kotlinx-serialization-json", version.ref = "kotlin-serialization" }
turbine = { group = "app.cash.turbine", name = "turbine", version.ref = "turbine" }

# app/build.gradle.kts
android {
    buildFeatures { compose = true }
    // No composeOptions block needed with Kotlin 2.0+ Compose compiler plugin
}
dependencies {
    val bom = platform(libs.compose.bom)
    implementation(bom)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui)
    implementation(libs.navigation.compose)
    implementation(libs.lottie.compose)
    implementation(libs.okhttp)
    implementation(libs.gson)
    implementation(libs.kotlinx.serialization.json)
    // Local JARs — preserved from old app
    implementation(files("libs/cae.jar"))
    implementation(files("libs/AlsaRecorder.jar"))
    testImplementation(libs.turbine)
}
```

---

## Architecture Patterns

### Recommended Project Structure
```
app/src/main/java/com/gow/smaitrobot/
├── JackieApplication.kt          # Application, DI graph root
├── MainActivity.kt               # Single activity, hosts NavHost
├── navigation/
│   └── AppNavigation.kt          # NavHost, BottomNavBar, sealed Screen routes
├── ui/
│   ├── theme/
│   │   ├── AppTheme.kt           # MaterialTheme wrapper, reads ThemeConfig
│   │   ├── ThemeConfig.kt        # data class parsed from JSON
│   │   └── WiEColors.kt          # Default WiE 2026 color tokens
│   ├── home/
│   │   ├── HomeScreen.kt
│   │   └── HomeViewModel.kt
│   ├── conversation/
│   │   ├── ConversationScreen.kt
│   │   ├── ConversationViewModel.kt
│   │   └── RobotAvatar.kt        # Lottie composable
│   ├── navigation_map/
│   │   ├── NavigationMapScreen.kt
│   │   └── NavigationMapViewModel.kt
│   ├── facilities/
│   │   ├── FacilitiesScreen.kt
│   │   └── FacilitiesViewModel.kt
│   ├── eventinfo/
│   │   ├── EventInfoScreen.kt
│   │   └── EventInfoViewModel.kt
│   └── common/
│       ├── SponsorBar.kt
│       ├── TopLogoBar.kt
│       └── FeedbackDialog.kt
├── data/
│   ├── websocket/
│   │   ├── WebSocketRepository.kt     # OkHttp3 client, SharedFlow emission
│   │   └── WebSocketEvent.kt         # sealed class: BinaryFrame, JsonMessage, Connected, Disconnected
│   ├── theme/
│   │   └── ThemeRepository.kt        # loads assets/theme.json → ThemeConfig StateFlow
│   └── model/
│       ├── ChatMessage.kt
│       ├── NavStatus.kt
│       └── PoiItem.kt
├── audio/
│   ├── CaeAudioManager.kt            # MOVED from old app, unchanged
│   └── TtsAudioPlayer.kt             # MOVED from old app, unchanged
assets/
├── wie2026_theme.json                # WiE 2026 event config
└── default_theme.json                # Fallback/default config
res/
├── raw/
│   ├── robot_idle.json               # Lottie animation JSON
│   ├── robot_listening.json
│   ├── robot_thinking.json
│   └── robot_speaking.json
```

### Pattern 1: WebSocket Repository with SharedFlow
**What:** WebSocket events emitted as a SharedFlow; ViewModels subscribe and transform to StateFlow
**When to use:** Any screen that needs real-time server data

```kotlin
// Source: https://medium.com/@danimahardhika/handle-websocket-in-jetpack-compose-with-okhttp-and-sharedflow-b1ed7c9fd713
class WebSocketRepository(private val client: OkHttpClient) {

    private val _events = MutableSharedFlow<WebSocketEvent>(extraBufferCapacity = 64)
    val events: SharedFlow<WebSocketEvent> = _events.asSharedFlow()

    private var webSocket: WebSocket? = null

    fun connect(url: String) {
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                _events.tryEmit(WebSocketEvent.BinaryFrame(bytes.toByteArray()))
            }
            override fun onMessage(webSocket: WebSocket, text: String) {
                _events.tryEmit(WebSocketEvent.JsonMessage(text))
            }
            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                _events.tryEmit(WebSocketEvent.Disconnected(t.message))
                scheduleReconnect()
            }
        })
    }

    fun send(bytes: ByteArray) { webSocket?.send(bytes.toByteString()) }
    fun send(json: String) { webSocket?.send(json) }
}

// ViewModel subscribes:
class ConversationViewModel(private val wsRepo: WebSocketRepository) : ViewModel() {
    private val _transcript = MutableStateFlow<List<ChatMessage>>(emptyList())
    val transcript: StateFlow<List<ChatMessage>> = _transcript.asStateFlow()

    init {
        viewModelScope.launch {
            wsRepo.events.collect { event ->
                when (event) {
                    is WebSocketEvent.JsonMessage -> handleJson(event.text)
                    is WebSocketEvent.BinaryFrame -> ttsPlayer.handleBinaryFrame(event.bytes)
                    else -> { /* connection state handling */ }
                }
            }
        }
    }
}
```

### Pattern 2: Runtime JSON Theming
**What:** `ThemeRepository` reads `assets/wie2026_theme.json` at startup, parses into `ThemeConfig`, exposes as `StateFlow`. `AppTheme` composable builds `ColorScheme` and `Typography` from config.
**When to use:** At app startup; any time theme needs to be swapped (different JSON file)

```kotlin
// Theme JSON structure (assets/wie2026_theme.json)
{
  "eventName": "WiE 2026",
  "tagline": "Engineering Beyond Imagination",
  "logoAsset": "wie_logo.png",
  "colors": {
    "primary": "#7B2D8B",
    "secondary": "#00A99D",
    "tertiary": "#F7941D",
    "background": "#FAFAFA",
    "onPrimary": "#FFFFFF"
  },
  "fonts": {
    "heading": "Montserrat",
    "body": "Roboto"
  },
  "sponsors": [
    { "name": "SJSU Engineering", "logoAsset": "sjsu_logo.png" }
  ],
  "cards": [
    { "label": "Ask Me Anything", "action": "navigate:chat", "icon": "chat" },
    { "label": "Guided Tour", "action": "navigate:map", "icon": "map" },
    { "label": "Keynote Info", "action": "inline:keynote", "icon": "star" },
    { "label": "Session Tracks", "action": "inline:sessions", "icon": "schedule" },
    { "label": "Facilities", "action": "navigate:facilities", "icon": "location" },
    { "label": "Event Info", "action": "navigate:eventinfo", "icon": "info" }
  ],
  "schedule": [...],
  "speakers": [...]
}

// ThemeRepository.kt
class ThemeRepository(private val context: Context) {
    private val _config = MutableStateFlow(ThemeConfig.default())
    val config: StateFlow<ThemeConfig> = _config.asStateFlow()

    fun load(assetFileName: String = "wie2026_theme.json") {
        val json = context.assets.open(assetFileName).bufferedReader().readText()
        _config.value = Gson().fromJson(json, ThemeConfig::class.java)
    }
}

// AppTheme.kt
@Composable
fun AppTheme(config: ThemeConfig, content: @Composable () -> Unit) {
    val colorScheme = lightColorScheme(
        primary = Color(android.graphics.Color.parseColor(config.colors.primary)),
        secondary = Color(android.graphics.Color.parseColor(config.colors.secondary)),
        tertiary = Color(android.graphics.Color.parseColor(config.colors.tertiary)),
        background = Color(android.graphics.Color.parseColor(config.colors.background))
    )
    MaterialTheme(colorScheme = colorScheme, content = content)
}
```

### Pattern 3: Type-Safe Navigation with Bottom Nav
**What:** Sealed class destinations, NavHost in MainActivity, NavigationBar always visible
**When to use:** All screen transitions

```kotlin
// Source: https://medium.com/androiddevelopers/type-safe-navigation-for-compose-105325a97657
@Serializable sealed class Screen {
    @Serializable object Home : Screen()
    @Serializable object Chat : Screen()
    @Serializable object Map : Screen()
    @Serializable object Facilities : Screen()
    @Serializable object EventInfo : Screen()
}

// MainActivity.kt
@Composable
fun AppScaffold(navController: NavHostController) {
    val currentEntry by navController.currentBackStackEntryAsState()
    Scaffold(
        bottomBar = {
            NavigationBar {
                screens.forEach { screen ->
                    NavigationBarItem(
                        selected = currentEntry?.destination?.hasRoute(screen::class) == true,
                        onClick = { navController.navigate(screen) {
                            popUpTo(Screen.Home) { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }},
                        icon = { /* icon */ },
                        label = { Text(screen.label) }
                    )
                }
            }
        }
    ) { padding ->
        NavHost(navController, startDestination = Screen.Home, Modifier.padding(padding)) {
            composable<Screen.Home> { HomeScreen(hiltViewModel()) }
            composable<Screen.Chat> { ConversationScreen(hiltViewModel()) }
            composable<Screen.Map> { NavigationMapScreen(hiltViewModel()) }
            composable<Screen.Facilities> { FacilitiesScreen(hiltViewModel()) }
            composable<Screen.EventInfo> { EventInfoScreen(hiltViewModel()) }
        }
    }
}
```

### Pattern 4: Lottie Robot Avatar with State Machine
**What:** 4 Lottie JSON files in `res/raw/`, composable switches animation based on `robotState`
**When to use:** ConversationScreen robot face

```kotlin
// Source: https://github.com/airbnb/lottie/blob/master/android-compose.md
enum class RobotState { IDLE, LISTENING, THINKING, SPEAKING }

@Composable
fun RobotAvatar(robotState: RobotState, modifier: Modifier = Modifier) {
    val animRes = when (robotState) {
        RobotState.IDLE -> R.raw.robot_idle
        RobotState.LISTENING -> R.raw.robot_listening
        RobotState.THINKING -> R.raw.robot_thinking
        RobotState.SPEAKING -> R.raw.robot_speaking
    }
    val composition by rememberLottieComposition(LottieCompositionSpec.RawRes(animRes))
    val progress by animateLottieCompositionAsState(
        composition,
        iterations = LottieConstants.IterateForever
    )
    LottieAnimation(composition, { progress }, modifier)
}
```

### Anti-Patterns to Avoid
- **Shared single ViewModel across all screens:** Each screen should have its own scoped ViewModel (`navBackStackEntry`); only `WebSocketRepository` is singleton
- **Collecting StateFlow in non-lifecycle-aware way:** Use `collectAsStateWithLifecycle()` not `collectAsState()` — prevents leaks when app is backgrounded
- **Binary protocol mutation:** Do NOT add framing layer; the 0x01-0x06 type byte prefix must stay exactly as the Python server sends it
- **Reading JSON on main thread at startup:** Call `ThemeRepository.load()` in a coroutine (`viewModelScope.launch(Dispatchers.IO)`)
- **Camera2 raw in Compose:** Wrap Camera2 in `AndroidView(TextureView)` or migrate to CameraX with `CameraXViewfinder` composable — avoid putting Camera2 directly in Compose tree

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Animated robot face | Custom Canvas drawing with frame timers | Lottie for Compose 6.x | Lottie handles timing, interpolation, state transitions; free robot assets on LottieFiles |
| WebSocket reconnect logic | Custom exponential backoff with Handler | Extract existing OkHttp3 pattern from old `MainActivity.kt` | Already tested and working; just move to repository |
| Theme color parsing | Manual hex→Color conversion utility | `android.graphics.Color.parseColor()` (built-in) | Standard Android API, handles all hex formats |
| JSON deserialization | Manual JSONObject field traversal | Gson with data classes | Type-safe, less error-prone for nested theme config |
| Chat auto-scroll | Custom RecyclerView-style scroll tracking | `LazyListState.animateScrollToItem()` triggered by `LaunchedEffect(messages.size)` | Single line in Compose |
| Bottom nav state restoration | Manual backstack management | NavHost `saveState = true` + `restoreState = true` | Built into Navigation Compose |
| Image display for map PNG | Custom Bitmap decode/draw cycle | `val bitmap = BitmapFactory.decodeByteArray(bytes, ...)` → `Image(bitmap.asImageBitmap())` | Two lines in Compose |

**Key insight:** The existing `CaeAudioManager.kt` and `TtsAudioPlayer.kt` are already architected as pure classes with injectable writers for testing — do NOT rewrite them. Move them verbatim and wrap in ViewModels.

---

## Common Pitfalls

### Pitfall 1: Compose Recomposition on Binary Frame Arrival
**What goes wrong:** WebSocket binary frames arrive at 100Hz (audio); if the ViewModel emits UI state on every frame, recomposition thrashes and audio glitches occur.
**Why it happens:** Forgetting that audio frames should NOT update UI state — only JSON messages and frame type 0x06 (map PNG) need UI updates.
**How to avoid:** Route binary frames in `WebSocketRepository` before they reach UI-emitting Flows: audio frames (0x01/0x03) → `CaeAudioManager`; TTS frames (0x05) → `TtsAudioPlayer`; map frames (0x06) → `MutableStateFlow<Bitmap?>` in MapViewModel. Only JSON messages → general event bus.
**Warning signs:** Dropped audio frames, jittery UI, high recomposition count in Layout Inspector.

### Pitfall 2: AudioTrack and Camera2 Thread Violations
**What goes wrong:** `TtsAudioPlayer` and `CaeAudioManager` use background `HandlerThread`; calling `.stop()` or `.release()` from a Compose coroutine scope on the wrong thread causes IllegalStateException.
**Why it happens:** Compose ViewModels use Dispatchers.Main; audio APIs require calling from the right thread.
**How to avoid:** Keep the `HandlerThread` internally; expose only `start()`, `stop()`, `release()` which internally post to the handler. This already works in the existing code — preserve the pattern.

### Pitfall 3: Theme JSON Missing Field Crashes
**What goes wrong:** `Gson().fromJson()` returns null for missing JSON fields; `Color(null.parseColor(...))` NPE crashes app on startup.
**Why it happens:** JSON config file has wrong key name or missing field.
**How to avoid:** All `ThemeConfig` fields must have default values. Use `data class ThemeConfig(val colors: ThemeColors = ThemeColors.default())`. Validate in `ThemeRepository.load()` and fall back to `ThemeConfig.default()` on any parse exception.

### Pitfall 4: Navigation Back Stack and Home Auto-Return
**What goes wrong:** 30-second auto-return to Home uses `navController.navigate(Screen.Home)` inside a coroutine, causing a crash if the Activity is in background.
**Why it happens:** NavController navigation must happen on the main thread while the Composable lifecycle is STARTED.
**How to avoid:** Use `navController.navigate(Screen.Home) { popUpTo(0) { inclusive = true } }` — and only trigger from a `LaunchedEffect` or `collectAsStateWithLifecycle()` observer that is lifecycle-aware. Let the ViewModel emit a `NavigateTo(Home)` event via `Channel<UiEvent>` and collect it in the Composable with lifecycle awareness.

### Pitfall 5: Immersive Mode with Compose
**What goes wrong:** Setting `WindowInsetsController` incorrectly in Compose leaves system bars visible on older Android versions (Jackie runs API 23+).
**Why it happens:** The old `SYSTEM_UI_FLAG_IMMERSIVE_STICKY` flags API is deprecated in API 30+ but required for API 23-29.
**How to avoid:** Use the compatibility shim:
```kotlin
// In MainActivity.onCreate():
WindowCompat.setDecorFitsSystemWindows(window, false)
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
    window.insetsController?.hide(WindowInsets.Type.systemBars())
    window.insetsController?.systemBarsBehavior =
        WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
} else {
    @Suppress("DEPRECATION")
    window.decorView.systemUiVisibility = (
        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or View.SYSTEM_UI_FLAG_FULLSCREEN
        or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
    )
}
```

### Pitfall 6: Lottie Robot Face Transition Jank
**What goes wrong:** Switching `robotState` causes the new Lottie composition to restart from frame 0 with a visible flash.
**Why it happens:** `rememberLottieComposition` with a new `RawRes` discards previous composition state instantly.
**How to avoid:** Preload all 4 compositions at `RobotAvatar` init time and use `AnimatedContent` with a `crossfade` transition spec between them. Or use a single Lottie file with multiple markers for each state.

---

## Code Examples

Verified patterns from official sources:

### Chat Auto-Scroll (LazyColumn + LaunchedEffect)
```kotlin
// Source: https://developer.android.com/develop/ui/compose/lists
@Composable
fun TranscriptList(messages: List<ChatMessage>) {
    val listState = rememberLazyListState()
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }
    LazyColumn(state = listState) {
        items(messages, key = { it.id }) { msg ->
            ChatBubble(msg)
        }
    }
}
```

### StateFlow Collection with Lifecycle Safety
```kotlin
// Source: https://developer.android.com/develop/ui/compose/architecture
@Composable
fun ConversationScreen(viewModel: ConversationViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    // uiState collection pauses when app is backgrounded — prevents memory leaks
}
```

### LazyVerticalGrid for Home Screen Cards
```kotlin
// Source: https://developer.android.com/develop/ui/compose/lists
@Composable
fun HomeCardGrid(cards: List<CardConfig>, onCardTap: (CardConfig) -> Unit) {
    LazyVerticalGrid(
        columns = GridCells.Fixed(3),  // 2x3 grid for 6 cards
        contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(cards) { card ->
            HomeCard(card = card, onClick = { onCardTap(card) })
        }
    }
}
```

### Map PNG Bitmap Display
```kotlin
// Pattern: decode 0x06 frame bytes → Bitmap → ImageBitmap → Image composable
val mapBitmap: StateFlow<ImageBitmap?> = wsRepo.events
    .filterIsInstance<WebSocketEvent.BinaryFrame>()
    .filter { it.bytes[0] == 0x06.toByte() }
    .map { BitmapFactory.decodeByteArray(it.bytes, 1, it.bytes.size - 1)?.asImageBitmap() }
    .stateIn(viewModelScope, SharingStarted.Eagerly, null)

// In Composable:
val bitmap by viewModel.mapBitmap.collectAsStateWithLifecycle()
bitmap?.let { Image(bitmap = it, contentDescription = "Map", modifier = Modifier.fillMaxSize()) }
```

### Sponsor Bar Horizontal Scroll
```kotlin
// Slow auto-scroll with LaunchedEffect animation
@Composable
fun SponsorBar(sponsors: List<SponsorConfig>) {
    val scrollState = rememberScrollState()
    LaunchedEffect(Unit) {
        while (true) {
            delay(50L)  // 20fps scroll cadence
            if (scrollState.value < scrollState.maxValue) {
                scrollState.scrollTo(scrollState.value + 1)
            } else {
                scrollState.scrollTo(0)  // loop
            }
        }
    }
    Row(
        Modifier
            .horizontalScroll(scrollState, enabled = false)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        sponsors.forEach { SponsorLogo(it) }
    }
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| XML layouts + ViewBinding | Jetpack Compose declarative UI | 2021 (stable), standard by 2023 | No RecyclerView adapters, no XML files |
| `SYSTEM_UI_FLAG_*` for immersive | `WindowInsetsController` API | API 30 (2020) | Needs compat shim for API 23+ |
| Camera2 raw in Activity | CameraX + `CameraXViewfinder` composable | 2025 (I/O '25, stable) | No `AndroidView(TextureView)` wrapper needed for camera |
| String-based navigation routes | Type-safe `@Serializable` routes | Navigation 2.8 (2024) | Compile-time safety, no route string typos |
| LiveData in ViewModel | StateFlow + `collectAsStateWithLifecycle()` | 2022-2023 | Kotlin-native, no Android dependency in domain layer |
| `composeOptions { kotlinCompilerExtensionVersion }` | Compose Compiler Gradle plugin (automatic with Kotlin 2.0+) | Kotlin 2.0 (2024) | Simpler build config |

**Deprecated/outdated:**
- `AppCompatActivity` XML theme inflation: replaced by `ComponentActivity` + Compose
- `RecyclerView.Adapter`: replaced by `LazyColumn`/`LazyVerticalGrid`
- `ViewBinding`/`DataBinding`: replaced by Compose state
- `LiveData<T>`: prefer `StateFlow<T>` for new code (LiveData still works but is legacy)

---

## Open Questions

1. **Lottie robot animation assets**
   - What we know: Lottie files need to be 3-4 JSON animations covering idle/listening/thinking/speaking states
   - What's unclear: Are there existing LottieFiles assets that look like a robot/avatar? Need to download from lottiefiles.com before implementation
   - Recommendation: Search LottieFiles for "robot", "assistant", or "avatar" and pick one with separate state markers, OR create simple abstract face animations in LottieFiles editor

2. **Camera2 vs CameraX for selfie capture**
   - What we know: CameraX with `CameraXViewfinder` went stable at Google I/O 2025 and works natively in Compose
   - What's unclear: Whether the RK3588 SoC on Jackie has any CameraX HAL compatibility issues (the existing Camera2 code had a bug with `LENS_FACING_EXTERNAL` that was fixed manually)
   - Recommendation: Start with CameraX `CameraXViewfinder` for Compose integration; if HAL issues arise on RK3588, fall back to `AndroidView(TextureView)` wrapping Camera2 (same as old pattern)

3. **Feedback local storage fallback**
   - What we know: If WebSocket disconnected when session ends, feedback must be saved locally and synced on reconnect
   - What's unclear: Whether a simple `SharedPreferences` queue is sufficient or if a Room database is needed for multi-session accumulation
   - Recommendation: SharedPreferences with a JSON array of pending feedback entries — Room is overkill for ~1 feedback record per session

4. **armeabi-v7a ABI restriction**
   - What we know: Existing build restricts `abiFilters` to `armeabi-v7a` (required for CAE SDK native libs)
   - What's unclear: Whether Jackie's RK3588 needs arm64-v8a or armeabi-v7a for CAE SDK
   - Recommendation: Preserve `abiFilters += "armeabi-v7a"` exactly as-is; do not change without testing on device

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | JUnit 4 + Robolectric 4.13 + Turbine 1.2 + Compose Test |
| Config file | `app/src/test/` (JVM unit tests) + `app/src/androidTest/` (instrumented) |
| Quick run command | `./gradlew :app:testDebugUnitTest` |
| Full suite command | `./gradlew :app:test :app:connectedDebugAndroidTest` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| APP-01 | App builds with Compose + MVVM | build | `./gradlew :app:assembleDebug` | Wave 0 |
| APP-02 | ThemeConfig parsed from JSON, colors applied | unit | `./gradlew test --tests "ThemeRepositoryTest"` | Wave 0 |
| APP-03 | HomeCardGrid renders N cards from config | unit (Robolectric) | `./gradlew test --tests "HomeScreenTest"` | Wave 0 |
| APP-04 | MapViewModel decodes 0x06 frame → ImageBitmap | unit | `./gradlew test --tests "NavigationMapViewModelTest"` | Wave 0 |
| APP-05 | Transcript list updates on JSON message; avatar state transitions | unit | `./gradlew test --tests "ConversationViewModelTest"` | Wave 0 |
| APP-06 | FacilitiesViewModel filters POIs; navigate_to sends correct WS message | unit | `./gradlew test --tests "FacilitiesViewModelTest"` | Wave 0 |
| APP-07 | EventInfoViewModel loads schedule from ThemeConfig | unit | `./gradlew test --tests "EventInfoViewModelTest"` | Wave 0 |
| APP-08 | TtsAudioPlayer routes 0x05 → AudioTrack (existing tests pass) | unit | `./gradlew test --tests "TtsAudioPlayerTest"` | EXISTS |
| APP-08 | CaeAudioManager frame format (existing tests pass) | unit | `./gradlew test --tests "CaeAudioManagerTest"` | EXISTS |
| APP-08 | WebSocketRepository emits events for binary + text frames | unit | `./gradlew test --tests "WebSocketRepositoryTest"` | Wave 0 |
| APP-09 | ThemeConfig.default() compiles; JSON swap produces different ColorScheme | unit | `./gradlew test --tests "ThemeRepositoryTest"` | Wave 0 |
| WIE-01 | wie2026_theme.json parses; primary color is WiE purple | unit | `./gradlew test --tests "WiEThemeTest"` | Wave 0 |
| WIE-02 | WiE cards[] array produces 6 expected card labels | unit | `./gradlew test --tests "WiEThemeTest"` | Wave 0 |

### Sampling Rate
- **Per task commit:** `./gradlew :app:testDebugUnitTest` (JVM only, ~15s)
- **Per wave merge:** `./gradlew :app:test` (all unit tests)
- **Phase gate:** Full suite green before `/gsd:verify-work` + manual build install on emulator or device

### Wave 0 Gaps
- [ ] `app/src/test/java/com/gow/smaitrobot/ThemeRepositoryTest.kt` — covers APP-02, APP-09, WIE-01, WIE-02
- [ ] `app/src/test/java/com/gow/smaitrobot/WebSocketRepositoryTest.kt` — covers APP-08 (new WS layer)
- [ ] `app/src/test/java/com/gow/smaitrobot/HomeScreenTest.kt` — covers APP-03 (Robolectric)
- [ ] `app/src/test/java/com/gow/smaitrobot/ConversationViewModelTest.kt` — covers APP-05
- [ ] `app/src/test/java/com/gow/smaitrobot/NavigationMapViewModelTest.kt` — covers APP-04
- [ ] `app/src/test/java/com/gow/smaitrobot/FacilitiesViewModelTest.kt` — covers APP-06
- [ ] `app/src/test/java/com/gow/smaitrobot/EventInfoViewModelTest.kt` — covers APP-07
- [ ] Robolectric setup: `testImplementation("org.robolectric:robolectric:4.13")` + `@RunWith(RobolectricTestRunner::class)` config
- [ ] Turbine setup: `testImplementation("app.cash.turbine:turbine:1.2.0")`
- [ ] Existing tests `TtsAudioPlayerTest.kt` and `CaeAudioManagerTest.kt` — ALREADY EXIST, will continue passing

---

## Sources

### Primary (HIGH confidence)
- [Android Developers — BOM to library version mapping](https://developer.android.com/develop/ui/compose/bom/bom-mapping) — confirmed BOM 2026.03.00, Material3 1.4.0, Compose runtime 1.10.5
- [Lottie GitHub — android-compose.md](https://github.com/airbnb/lottie/blob/master/android-compose.md) — Compose API `LottieCompositionSpec.RawRes`, `animateLottieCompositionAsState`
- [Android Developers — Type safety in Navigation](https://developer.android.com/guide/navigation/design/type-safety) — `@Serializable` route objects, Navigation 2.8 API
- [Android Developers — Material Design 3 in Compose](https://developer.android.com/develop/ui/compose/designsystems/material3) — `lightColorScheme()`, `MaterialTheme`, Typography system
- [Android Developers — Compose architecture](https://developer.android.com/develop/ui/compose/architecture) — unidirectional data flow, `collectAsStateWithLifecycle()`
- Existing app source: `CaeAudioManager.kt`, `TtsAudioPlayer.kt`, `TtsAudioPlayerTest.kt`, `CaeAudioManagerTest.kt`, `AndroidManifest.xml`, `build.gradle.kts` — direct inspection

### Secondary (MEDIUM confidence)
- [ProAndroidDev — CameraX goes full Compose (Oct 2025)](https://proandroiddev.com/goodbye-androidview-camerax-goes-full-compose-4d21ca234c4e) — `CameraXViewfinder` composable, stable at I/O 2025
- [Medium — Handle WebSocket in Compose with OkHttp and SharedFlow](https://medium.com/@danimahardhika/handle-websocket-in-jetpack-compose-with-okhttp-and-sharedflow-b1ed7c9fd713) — SharedFlow pattern verified against official StateFlow docs
- [Medium — Type-Safe Navigation for Compose](https://medium.com/androiddevelopers/type-safe-navigation-for-compose-105325a97657) — authored by Don Turner (Android DevRel), HIGH confidence

### Tertiary (LOW confidence)
- UI/UX Pro Max skill — Jetpack Compose stack guidelines (`animate*AsState`, `mutableStateOf`) — alignment with project conventions
- WebSearch general results about Robolectric + Compose testing — cross-verified with Robolectric official site

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — BOM version verified via official BOM mapping page; existing app dependencies inspected directly
- Architecture: HIGH — MVVM + StateFlow + Navigation Compose patterns from official Android docs
- Pitfalls: MEDIUM — Recomposition/threading pitfalls from direct code inspection of existing app + general Compose knowledge; immersive mode compat from Android docs
- Animation (Lottie): MEDIUM — Lottie Compose API verified via GitHub official docs; specific robot animation assets are LOW (must be acquired)
- CameraX: MEDIUM — Stable status confirmed from ProAndroidDev article (Oct 2025); RK3588 compatibility is LOW (unknown until tested on device)

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable ecosystem — 30-day validity reasonable)
