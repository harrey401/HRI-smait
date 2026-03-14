# Phase 11: Wayfinding LLM Tools and Display Rendering (HOME) — Research

**Researched:** 2026-03-14
**Domain:** OpenAI function-calling tool registration, PIL map overlay rendering with destination highlight, display dispatch via ConnectionManager, EventBus-driven verbal status updates
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| WAY-01 | LLM has tool/function for querying location database ("where is X?") | OpenAI `tools` parameter with `query_location` function definition; handler calls `POIKnowledgeBase.resolve()` and `fetch_markers()` |
| WAY-02 | LLM has tool/function for initiating navigation ("take me to X") | `navigate_to` function definition; handler calls `NavController.navigate_to()` and returns verbal confirmation |
| WAY-03 | When user asks "where is X?", system shows map with highlighted path on touchscreen + gives verbal directions | `MapManager.render_map()` extended with destination highlight circle overlay; bytes dispatched via new `DISPLAY_MAP` EventType → `ConnectionManager.send_map_image()` |
| WAY-04 | When user says "take me to X", robot navigates and converses during transit | `navigate_to` tool fires verbal status update; `NAV_ARRIVED` / `NAV_FAILED` EventBus events trigger TTS verbal responses and display status update |
| WAY-05 | Robot verbally confirms arrival or explains navigation failure | Subscribe to `NAV_ARRIVED` and `NAV_FAILED` in `WayfindingManager`; emit `DIALOGUE_RESPONSE` with verbal text → triggers TTS pipeline |
| DISP-01 | Map with robot position, POIs, and path rendered and sent to Jackie touchscreen via WebSocket | `ConnectionManager.send_map_image()` sends PNG bytes as binary frame (new frame type 0x06) + JSON metadata; triggered by `DISPLAY_MAP` event |
| DISP-02 | Navigation status shown on touchscreen (navigating to X, arrived) | `ConnectionManager.send_nav_status()` sends JSON `{"type": "nav_status", "status": "navigating", "destination": "ENG192"}`; triggered by new `DISPLAY_NAV_STATUS` EventType |
</phase_requirements>

---

## Summary

Phase 11 builds the LLM tool-use layer on top of the Phase 10 spatial primitives (MapManager, POIKnowledgeBase, NavController) and the existing dialogue manager (DialogueManager). Two LLM tools are registered: `query_location` and `navigate_to`. These tools are standard OpenAI function-calling definitions; the existing `DialogueManager.ask()` is extended to pass `tools=` and handle `tool_calls` in the response.

The display rendering pipeline adds destination highlighting to `MapManager.render_map()`, then dispatches rendered map PNG bytes via the EventBus to `ConnectionManager`, which sends them to Jackie's Android app over WebSocket. A new binary frame type (0x06) carries the map image; a JSON text message carries navigation status. Both are consumed by the Phase 12 Android app.

The verbal response flow uses the existing TTS pipeline: `WayfindingManager` subscribes to `NAV_ARRIVED` / `NAV_FAILED` and emits `DIALOGUE_RESPONSE` events with human-readable text — these automatically flow through `TTSEngine` and play on Jackie's speaker via the established 0x05 pipeline.

**Primary recommendation:** Introduce a single new class `WayfindingManager` (`smait/navigation/wayfinding_manager.py`) that owns tool registration, tool dispatch, display dispatch, and arrival/failure verbal responses. Extend `DialogueManager` minimally (pass `tools` and handle `tool_calls`). Extend `ConnectionManager` with map/status send methods. Add two new EventTypes: `DISPLAY_MAP` and `DISPLAY_NAV_STATUS`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `openai` | 2.21.0 (verified in venv) | Tool-use function calling with `tools=` parameter | Already installed; `gpt-4o-mini` supports function calling |
| `Pillow` (PIL) | 11.3.0 (verified in venv) | Destination circle overlay on rendered map | Already used by MapManager for robot arrow; same ImageDraw API |
| `json` | stdlib | Tool argument parsing, nav status JSON messages | No install needed |
| `asyncio` | stdlib | Async tool handler dispatch | All handlers are async |
| `dataclasses` | stdlib | Tool result dataclass (POI match, nav confirmation) | Follows Config pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `websockets` | 16.0 (verified in venv) | ConnectionManager display send | Already in use; no new dep |
| `base64` | stdlib | Not needed for display send (raw PNG bytes) | Only if Android expects base64 JSON payload |
| `unittest.mock` | stdlib | Mock ChassisClient, NavController, ConnectionManager in tests | Standard test isolation |
| `pytest-asyncio` | installed | Async test support | Already used by test suite |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| OpenAI function calling | Ollama tool calling | Ollama tool-use requires specific model support (not all models); OpenAI gpt-4o-mini is the reliable fallback path in this codebase — use OpenAI format first |
| New binary frame (0x06) for map | JSON base64 payload | Binary is more efficient for PNG (maps are 50-200KB); consistent with audio 0x01/0x05 pattern |
| `WayfindingManager` as new class | Inline in DialogueManager | DialogueManager is already complex; clean separation of concerns; Phase 12 app depends on the display protocol being stable |

**Installation:** No new packages needed. All dependencies are already installed in the venv.

---

## Architecture Patterns

### Recommended Project Structure
```
smait/
├── navigation/
│   ├── __init__.py              # ADD WayfindingManager to exports
│   ├── map_manager.py           # EXTEND: add highlight_destination() rendering
│   ├── poi_knowledge_base.py    # NO CHANGE — used by WayfindingManager
│   ├── nav_controller.py        # NO CHANGE — called by WayfindingManager
│   └── wayfinding_manager.py    # NEW: tool registration + dispatch + display
├── dialogue/
│   └── manager.py               # EXTEND: add register_tools(), handle tool_calls
├── connection/
│   ├── manager.py               # EXTEND: send_map_image(), send_nav_status()
│   └── protocol.py              # EXTEND: FrameType.MAP_IMAGE = 0x06, MessageSchema methods
├── core/
│   └── events.py                # ADD: DISPLAY_MAP, DISPLAY_NAV_STATUS EventTypes

tests/
└── unit/
    ├── test_wayfinding_manager.py  # NEW — WAY-01 to WAY-05
    └── test_display_dispatch.py    # NEW — DISP-01, DISP-02
```

### Pattern 1: OpenAI Tool Registration and Dispatch

**What:** `WayfindingManager.register_tools(dialogue_manager)` passes a list of tool definitions to `DialogueManager`. On each `ask()` call, if the LLM returns `tool_calls`, the dialogue manager invokes the registered handlers and feeds results back into the conversation.

**When to use:** Any time user asks a wayfinding question — handled transparently inside `ask()`.

**How OpenAI tool calling works (verified against openai 2.x API):**

```python
# Tool definitions passed to openai.chat.completions.create
WAYFINDING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_location",
            "description": "Find where a named location is on the current floor map. Use when user asks 'where is X?' or 'how do I get to X?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": "The location name as spoken by the user (e.g. 'ENG192', 'the bathroom', 'registration desk')"
                    }
                },
                "required": ["location_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Start navigating the robot to a named location. Use when user says 'take me to X', 'go to X', or 'navigate to X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {
                        "type": "string",
                        "description": "The resolved POI name (use after query_location confirms it exists)"
                    }
                },
                "required": ["poi_name"]
            }
        }
    }
]
```

**Tool call response handling:**

```python
# In DialogueManager._ask_api() — extended for tool calls
response = await self._openai_client.chat.completions.create(
    model=self._config.api_model,
    messages=messages,
    tools=self._tools,          # injected by WayfindingManager
    tool_choice="auto",
    max_tokens=self._config.max_tokens,
)

message = response.choices[0].message
if message.tool_calls:
    # Execute tool, inject result, call LLM again
    for tool_call in message.tool_calls:
        result = await self._tool_handlers[tool_call.function.name](
            json.loads(tool_call.function.arguments)
        )
        # Append tool result to messages and re-invoke LLM
```

**Key detail:** Tool calling requires two LLM round-trips when tools are invoked: one to get the tool call, one to generate the verbal response after the tool result is injected. The existing `_ask_api()` flow handles this with a follow-up call.

### Pattern 2: WayfindingManager Tool Handler Flow

**`query_location` handler:**
1. Call `POIKnowledgeBase.resolve(location_name)` → get chassis marker name
2. If found: get POI coordinates from `chassis_markers` cache
3. Call `MapManager.render_map_with_highlight(poi_name, px, py)` → PNG bytes
4. Emit `DISPLAY_MAP` event with PNG bytes
5. Return `{"found": True, "poi_name": marker_name, "verbal": "ENG192 is on the third floor..."}`

**`navigate_to` handler:**
1. Call `NavController.navigate_to(poi_name)` (async)
2. If nav started: return `{"started": True, "verbal": "On my way to ENG192!"}`
3. If nav failed: return `{"started": False, "verbal": "Sorry, I couldn't find a path to ENG192."}`

```python
# smait/navigation/wayfinding_manager.py
class WayfindingManager:
    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        poi_kb: POIKnowledgeBase,
        nav_controller: NavController,
        map_manager: MapManager,
    ) -> None:
        self._cfg = config
        self._bus = event_bus
        self._poi_kb = poi_kb
        self._nav = nav_controller
        self._map = map_manager

        # Subscribe to nav outcome events for verbal confirmations
        self._bus.subscribe(EventType.NAV_ARRIVED, self._on_nav_arrived)
        self._bus.subscribe(EventType.NAV_FAILED, self._on_nav_failed)

    def get_tools(self) -> list[dict]:
        """Return OpenAI tool definitions for registration with DialogueManager."""
        return WAYFINDING_TOOLS

    def get_tool_handlers(self) -> dict[str, Callable]:
        """Return name → async handler mapping."""
        return {
            "query_location": self._handle_query_location,
            "navigate_to": self._handle_navigate_to,
        }

    async def _handle_query_location(self, args: dict) -> dict:
        location_name = args.get("location_name", "")
        poi_name = self._poi_kb.resolve(location_name)
        if poi_name is None:
            known = self._poi_kb.list_locations()
            return {"found": False, "verbal": f"I don't know where {location_name} is. I know about: {', '.join(known[:5])}."}
        # Render map with highlight and dispatch
        png_bytes = self._map.render_map_with_highlight(poi_name)
        self._bus.emit(EventType.DISPLAY_MAP, {"png": png_bytes, "highlighted_poi": poi_name})
        return {"found": True, "poi_name": poi_name, "verbal": f"I found {location_name} on the map!"}

    async def _handle_navigate_to(self, args: dict) -> dict:
        poi_name = args.get("poi_name", "")
        result = await self._nav.navigate_to(poi_name)
        if result.get("success"):
            # Emit nav status display update
            self._bus.emit(EventType.DISPLAY_NAV_STATUS, {"status": "navigating", "destination": poi_name})
            return {"started": True, "verbal": f"On my way to {poi_name}!"}
        return {"started": False, "verbal": f"Sorry, I couldn't navigate to {poi_name}."}

    def _on_nav_arrived(self, data: dict) -> None:
        destination = data.get("destination", "the destination")
        self._bus.emit(EventType.DISPLAY_NAV_STATUS, {"status": "arrived", "destination": destination})
        self._bus.emit(EventType.DIALOGUE_RESPONSE, _make_verbal_response(
            f"We've arrived at {destination}!"
        ))

    def _on_nav_failed(self, data: dict) -> None:
        destination = data.get("destination", "the destination")
        reason = data.get("reason", "unknown")
        self._bus.emit(EventType.DISPLAY_NAV_STATUS, {"status": "failed", "destination": destination})
        self._bus.emit(EventType.DIALOGUE_RESPONSE, _make_verbal_response(
            f"Sorry, I wasn't able to reach {destination}. {_reason_to_verbal(reason)}"
        ))
```

### Pattern 3: MapManager Destination Highlight Rendering

**What:** Extend `MapManager.render_map()` to accept an optional highlighted POI name and draw a colored circle at the POI pixel position.

**Key detail:** The highlight uses world-to-pixel transform already in `map_manager.py`. POI coordinates come from the `_chassis_markers` cache in `POIKnowledgeBase` (each marker has `pose.position.{x,y}`).

```python
# In map_manager.py — add render_map_with_highlight()
def render_map_with_highlight(
    self,
    poi_name: str,
    highlight_color: str = "yellow",
    radius: int = 12,
) -> bytes:
    """Render map with destination POI highlighted as a circle overlay.

    Args:
        poi_name: Chassis marker name whose position should be highlighted.
        highlight_color: PIL color string for the highlight circle.
        radius: Highlight circle radius in pixels.

    Returns:
        PNG image bytes with robot arrow, path, and destination highlight.
    """
    # Get POI world coordinates from cached markers
    poi_pos = self._poi_positions.get(poi_name)  # {"x": float, "y": float}
    png_bytes = self._render_with_extras(
        highlight_pos=poi_pos,
        highlight_color=highlight_color,
        highlight_radius=radius,
    )
    return png_bytes
```

**Internal detail:** `MapManager` needs a `_poi_positions: dict[str, dict]` cache (populated when `POI_LIST_UPDATED` event fires). This avoids tight coupling between `WayfindingManager` and the internal marker data.

### Pattern 4: Display Dispatch via ConnectionManager

**What:** `ConnectionManager` subscribes to `DISPLAY_MAP` and `DISPLAY_NAV_STATUS` events and sends them to Jackie over WebSocket.

**Frame type allocation:**
- `0x06` = MAP_IMAGE binary frame (PNG bytes — consistent with existing binary frame pattern)
- JSON text message `{"type": "nav_status", "status": "...", "destination": "..."}` — consistent with existing text frame pattern

```python
# In connection/protocol.py — add FrameType
class FrameType(IntEnum):
    AUDIO_CAE = 0x01
    VIDEO = 0x02
    AUDIO_RAW = 0x03
    CONTROL = 0x04
    TTS_AUDIO = 0x05
    MAP_IMAGE = 0x06       # NEW — PNG map image to Jackie touchscreen

# Add to MessageSchema
@staticmethod
def nav_status(status: str, destination: str) -> str:
    return json.dumps({"type": "nav_status", "status": status, "destination": destination})

@staticmethod
def map_image_metadata(width: int, height: int, highlighted_poi: str) -> str:
    return json.dumps({"type": "map_meta", "width": width, "height": height, "poi": highlighted_poi})
```

```python
# In connection/manager.py — add event subscriptions and send methods
# In __init__:
event_bus.subscribe(EventType.DISPLAY_MAP, self._on_display_map)
event_bus.subscribe(EventType.DISPLAY_NAV_STATUS, self._on_display_nav_status)

async def send_map_image(self, png_bytes: bytes) -> None:
    """Send map image to Jackie as binary frame 0x06."""
    frame = BinaryFrame.pack(FrameType.MAP_IMAGE, png_bytes)
    await self.send_binary(frame)

async def send_nav_status(self, status: str, destination: str) -> None:
    """Send navigation status update to Jackie as JSON text."""
    await self.send_text(MessageSchema.nav_status(status, destination))

async def _on_display_map(self, data: dict) -> None:
    png_bytes = data.get("png", b"")
    if png_bytes:
        await self.send_map_image(png_bytes)

async def _on_display_nav_status(self, data: dict) -> None:
    status = data.get("status", "unknown")
    destination = data.get("destination", "")
    await self.send_nav_status(status, destination)
```

### Pattern 5: DialogueManager Tool Registration Extension

**What:** Add `register_tools()` and `_tool_handlers` to `DialogueManager`. The `_ask_api()` method is extended to handle `tool_calls` in the response.

**Critical detail — two-round-trip tool call flow:**
```
User: "where is ENG192?"
→ LLM call with tools → tool_calls: [query_location(location_name="ENG192")]
→ Execute tool handler → returns {"found": True, "poi_name": "eng192", ...}
→ LLM call again with tool result → generates verbal response
→ Return verbal response to user → TTS pipeline
```

```python
# In dialogue/manager.py — minimal extension
def register_tools(self, tools: list[dict], handlers: dict[str, Callable]) -> None:
    """Register LLM tool definitions and async handler callables.

    Args:
        tools: OpenAI-format tool definition list.
        handlers: Dict mapping function name to async callable.
    """
    self._tools = tools
    self._tool_handlers = handlers

async def _ask_api(self, user_text: str) -> Optional[DialogueResponse]:
    """Extended: handles tool_calls in response."""
    messages = self._build_messages()
    kwargs = {"model": ..., "messages": messages, "max_tokens": ..., "temperature": ...}
    if self._tools:
        kwargs["tools"] = self._tools
        kwargs["tool_choice"] = "auto"

    response = await self._openai_client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    if message.tool_calls:
        return await self._handle_tool_calls(message, messages)
    # ... existing text response path
```

### Anti-Patterns to Avoid

- **Embedding navigation logic in DialogueManager:** DialogueManager is the LLM interface only. Tool execution belongs in WayfindingManager. Only `register_tools()` + `_handle_tool_calls()` are added to DialogueManager.
- **Synchronous PIL rendering in event handler:** `_on_map_update` is sync (called from EventBus). `render_map_with_highlight()` must also be sync (PIL is CPU-bound, fast enough for ~100ms maps). Do NOT make it async.
- **Storing PNG bytes on EventBus data dict by reference and mutating them:** Follow the existing immutable pattern — always copy the rendered bytes before emitting.
- **Adding WayfindingManager to `HRISystem.__init__` before Phase 11 is complete:** Phase 11 is HOME work with mocks only. Integration into `main.py` happens in Phase 13.
- **Using Ollama for tool calling:** Ollama's `phi-4-mini` / `mistral-nemo` support for tools is unreliable. Tool calling MUST use the OpenAI API path only. Fall back to text if OpenAI is unavailable (graceful degradation).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM function calling protocol | Custom JSON parsing of intent | OpenAI `tools` parameter | OpenAI handles argument parsing, type checking, retry — don't replicate |
| POI fuzzy name matching | Levenshtein distance / regex | `POIKnowledgeBase.resolve()` already exists from Phase 10 | Already handles case-insensitive lookup + chassis marker fallback |
| Map coordinate transform | Custom pixel math | `world_to_pixel()` from `map_manager.py` already exists | Tested, handles ROS Y-flip convention |
| Verbal response generation for arrival | Hardcoded string templates | LLM-generated (tool result injected, LLM makes natural response) | More natural; handles edge cases ("failed because obstacle") |
| Binary frame encoding | Custom byte packing | `BinaryFrame.pack(FrameType.MAP_IMAGE, png_bytes)` from Phase 9 | Consistent with existing protocol |

**Key insight:** Phase 11 primarily wires existing pieces together. The complexity is in the tool-call loop (two LLM round-trips) and the POI coordinate cache in MapManager. Both are manageable.

---

## Common Pitfalls

### Pitfall 1: Tool Calls on the Ollama Path
**What goes wrong:** Sending `tools=` to Ollama with a model that doesn't support function calling causes a 400 error or silent tool ignore.
**Why it happens:** Not all Ollama models support OpenAI-compatible function calling. `phi-4-mini` and `mistral-nemo` are hit-or-miss.
**How to avoid:** Gate tool_calling to OpenAI path only. In `_ask_ollama()`, do NOT pass tools. If Ollama is primary and tools are needed, fall through to OpenAI API path.
**Warning signs:** Ollama returns 400 or the LLM ignores the wayfinding question entirely.

### Pitfall 2: Missing Tool Result Injection (Half-Baked Two-Round-Trip)
**What goes wrong:** After tool execution, the tool result is not appended to the messages list before the second LLM call. The LLM hallucinates a response without the actual POI data.
**Why it happens:** OpenAI requires a specific message structure: assistant message with `tool_calls` + `tool` role message with `tool_call_id` and `content` = tool result JSON.
**How to avoid:**
```python
messages.append({"role": "assistant", "content": None, "tool_calls": message.tool_calls})
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(tool_result),
})
# Then make second LLM call with extended messages
```
**Warning signs:** LLM responds with "I found the location" without having real coordinates.

### Pitfall 3: POI Coordinate Cache Miss
**What goes wrong:** `render_map_with_highlight()` is called with a POI name but `MapManager._poi_positions` has no entry for it, so no highlight is drawn.
**Why it happens:** `POI_LIST_UPDATED` event fires when markers are fetched, but `MapManager` may not be subscribed to it yet, or the highlight code looks up by marker name while the cache stores by `name` key from the fetch response.
**How to avoid:** Subscribe `MapManager` to `POI_LIST_UPDATED` in `start()`. Cache `{marker["name"]: {"x": pose_x, "y": pose_y}}`. Draw highlight only when coordinates are found; skip silently otherwise.
**Warning signs:** Map rendered without any highlight circle even though POI exists.

### Pitfall 4: Verbal Response Event Type Collision
**What goes wrong:** `WayfindingManager._on_nav_arrived()` emits `DIALOGUE_RESPONSE` with a `DialogueResponse` object. If `TTSEngine` is subscribed to `DIALOGUE_RESPONSE` but expects a specific field (`text`), a mismatched payload causes silence or exceptions.
**Why it happens:** `DialogueResponse` is a dataclass; `TTSEngine` may check `data.text` directly.
**How to avoid:** Use the existing `DialogueResponse` dataclass from `smait/dialogue/manager.py` when emitting `DIALOGUE_RESPONSE`. Import and construct it properly in `WayfindingManager`.

### Pitfall 5: PNG Size and WebSocket Frame Limit
**What goes wrong:** Map PNG is large (>1MB for high-res occupancy grids) and exceeds the WebSocket `max_size=2**22` (4MB) limit in `ConnectionManager`.
**Why it happens:** Occupancy grid maps can be large. JPEG wouldn't help much since these are mostly grey pixels.
**How to avoid:** Resize map to max 800x800 before encoding to PNG in `render_map_with_highlight()`. Or convert to JPEG (smaller than PNG for photorealistic content). The map is already capped at display resolution — verify the rendering target size in `MapManager`.
**Warning signs:** WebSocket `ConnectionClosedError` on map send with large floors.

### Pitfall 6: Navigation Tool Called Without Resolved POI
**What goes wrong:** LLM calls `navigate_to(poi_name="room 192")` with the raw user text instead of the chassis marker name (`"eng192"`). `NavController.navigate_to()` resolves via `POIKnowledgeBase`, but if the LLM skips `query_location`, the fallback may fail silently.
**Why it happens:** LLM function calling is probabilistic — it may call `navigate_to` directly even if you intended `query_location` first.
**How to avoid:** Design `navigate_to` tool description to say "use after query_location confirms the location exists." Also make `NavController.navigate_to()` robust — it already does `resolve() or poi_name` fallback, so direct calls with human names often work.

---

## Code Examples

Verified patterns from existing codebase and openai 2.x API:

### OpenAI Tool Call Response Handling
```python
# Source: openai 2.21.0 API (verified in venv)
response = await self._openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=WAYFINDING_TOOLS,
    tool_choice="auto",
    max_tokens=150,
)
message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    name = tool_call.function.name          # "query_location"
    args = json.loads(tool_call.function.arguments)  # {"location_name": "ENG192"}
    handler = self._tool_handlers[name]
    result = await handler(args)            # returns dict
    # Inject into messages for second call:
    messages.append({"role": "assistant", "content": None,
                     "tool_calls": [tc.model_dump() for tc in message.tool_calls]})
    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                     "content": json.dumps(result)})
    # Second call (no tools this time — get verbal response)
    second_resp = await self._openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
    )
    return second_resp.choices[0].message.content
```

### MapManager Destination Highlight
```python
# Extension to smait/navigation/map_manager.py
# Source: PIL 11.3.0 ImageDraw.ellipse() API (verified in venv)
def render_map_with_highlight(self, poi_name: str) -> bytes:
    """Render map with destination highlighted as a filled circle."""
    if self._map_image is not None:
        base = self._map_image.copy()
        meta = self._map_meta
    else:
        base = Image.new("RGBA", _PLACEHOLDER_SIZE, color=(200, 200, 200, 255))
        meta = {"origin_x": 0.0, "origin_y": 0.0, "resolution": 1.0,
                "width": _PLACEHOLDER_SIZE[0], "height": _PLACEHOLDER_SIZE[1]}

    draw = ImageDraw.Draw(base)

    # Draw destination highlight
    poi_pos = self._poi_positions.get(poi_name)
    if poi_pos:
        hx, hy = world_to_pixel(poi_pos["x"], poi_pos["y"], meta)
        r = self._cfg.highlight_radius_px  # e.g. 12
        draw.ellipse([hx - r, hy - r, hx + r, hy + r],
                     fill=self._cfg.highlight_color,  # e.g. "yellow"
                     outline="orange", width=2)

    # Draw path and robot arrow (existing logic)
    if len(self._path_points) >= 2:
        pixel_path = [world_to_pixel(x, y, meta) for x, y in self._path_points]
        draw.line(pixel_path, fill=self._cfg.path_color, width=2)
    px, py = world_to_pixel(self._current_pose["x"], self._current_pose["y"], meta)
    draw_robot_arrow(draw, px, py, self._current_pose.get("theta", 0.0),
                     length=self._cfg.arrow_length_px, color=self._cfg.arrow_color)

    buf = io.BytesIO()
    base.save(buf, format="PNG")
    return buf.getvalue()
```

### EventType Additions (core/events.py)
```python
# Add to EventType enum after Phase 10 entries
DISPLAY_MAP = auto()              # data: {"png": bytes, "highlighted_poi": str}
DISPLAY_NAV_STATUS = auto()       # data: {"status": str, "destination": str}
```

### NavigationConfig Additions (core/config.py)
```python
# Add to NavigationConfig dataclass
highlight_color: str = "yellow"
highlight_radius_px: int = 12
```

### BinaryFrame extension (connection/protocol.py)
```python
# Add to FrameType IntEnum
MAP_IMAGE = 0x06    # Map PNG to Jackie touchscreen (PC → Jackie)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded intent detection ("if 'where' in text") | LLM function calling with structured tool definitions | openai 1.x (2023) | Handles natural language variations automatically |
| Separate intent detection model | OpenAI gpt-4o-mini handles both conversation and tool routing | 2024 | No extra model needed; gpt-4o-mini supports function calling reliably |
| Polling for navigation result | EventBus NAV_ARRIVED/NAV_FAILED events (Phase 10) | Phase 10 (2026) | Already done — subscribe and react |

**Deprecated/outdated:**
- `openai.ChatCompletion.create()` (v0.x style): Replaced by `openai.chat.completions.create()` in openai 1.x+. The codebase already uses the new API in `DialogueManager._ask_api()`.
- `functions=` parameter: Replaced by `tools=` in openai 1.x+. Use `tools` with `{"type": "function", "function": {...}}` wrapper.

---

## Open Questions

1. **Should `query_location` also send the robot to show the path (WAY-03)?**
   - What we know: WAY-03 says "shows map with highlighted path" — this implies path rendering, but a path requires navigation to have started.
   - What's unclear: Is "highlighted path" a straight-line visual preview from robot to POI, or the actual planned path (which only exists after `navigate_to`)?
   - Recommendation: Implement as straight-line preview (draw line from robot pose to POI position) for `query_location`. The actual planned path from `/global_path` only appears after `navigate_to` fires. This satisfies WAY-03 without requiring navigation to start.

2. **Does the Phase 12 Android app need to know how to decode 0x06 MAP_IMAGE frames?**
   - What we know: Phase 12 is the full Android app rewrite (separate phase). DISP-01/02 are Phase 11 requirements verified with mock tests.
   - What's unclear: Whether Phase 12 will need changes to accept 0x06 frames before Phase 11 is "done."
   - Recommendation: Phase 11 verifies display dispatch with mock ConnectionManager in unit tests. Phase 12 implements the Android receiver. DISP-01 success criteria is "dispatched to display channel and verified to contain correct visual overlay using test map fixture" — this can be fully verified without real Android app.

3. **Tool calling for Ollama path: degrade gracefully?**
   - What we know: Ollama is `try_local_first` but unreliable for tool use.
   - Recommendation: When `_tools` is non-empty and Ollama is attempted, check if the response contains `tool_calls`. If Ollama doesn't return tool_calls but the user's text likely needs tools (heuristic: contains "where" / "take me"), fall through to OpenAI API. This matches the existing fallback pattern.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (installed in venv) + pytest-asyncio |
| Config file | `pyproject.toml` (project root) |
| Quick run command | `source venv/bin/activate && python -m pytest tests/unit/test_wayfinding_manager.py tests/unit/test_display_dispatch.py -v` |
| Full suite command | `source venv/bin/activate && python -m pytest tests/unit/ -v --cov=smait --cov-report=term-missing` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WAY-01 | `query_location` tool resolves POI name via POIKnowledgeBase and returns match + coordinates | unit | `pytest tests/unit/test_wayfinding_manager.py::test_query_location_found -x` | ❌ Wave 0 |
| WAY-01 | `query_location` returns "not found" for unknown location | unit | `pytest tests/unit/test_wayfinding_manager.py::test_query_location_not_found -x` | ❌ Wave 0 |
| WAY-01 | `query_location` tool is registered in DialogueManager after `register_tools()` | unit | `pytest tests/unit/test_wayfinding_manager.py::test_tool_registration -x` | ❌ Wave 0 |
| WAY-02 | `navigate_to` tool calls `NavController.navigate_to()` and returns verbal confirmation | unit | `pytest tests/unit/test_wayfinding_manager.py::test_navigate_to_success -x` | ❌ Wave 0 |
| WAY-02 | `navigate_to` returns failure verbal when NavController returns no success | unit | `pytest tests/unit/test_wayfinding_manager.py::test_navigate_to_failure -x` | ❌ Wave 0 |
| WAY-02 | DialogueManager._ask_api() handles tool_calls via two-round-trip flow | unit | `pytest tests/unit/test_wayfinding_manager.py::test_dialogue_tool_call_flow -x` | ❌ Wave 0 |
| WAY-03 | `query_location` handler emits `DISPLAY_MAP` with PNG bytes containing highlight circle | unit | `pytest tests/unit/test_wayfinding_manager.py::test_query_location_dispatches_map -x` | ❌ Wave 0 |
| WAY-03 | `render_map_with_highlight()` draws circle at POI pixel position (verified with test fixture) | unit | `pytest tests/unit/test_wayfinding_manager.py::test_render_map_highlight_pixel -x` | ❌ Wave 0 |
| WAY-04 | `navigate_to` tool emits DISPLAY_NAV_STATUS with status="navigating" | unit | `pytest tests/unit/test_wayfinding_manager.py::test_navigate_to_dispatches_status -x` | ❌ Wave 0 |
| WAY-04 | Verbal status update is emitted to DIALOGUE_RESPONSE when navigate_to fires | unit | `pytest tests/unit/test_wayfinding_manager.py::test_navigate_to_verbal_status -x` | ❌ Wave 0 |
| WAY-05 | NAV_ARRIVED event triggers verbal confirmation and DISPLAY_NAV_STATUS update | unit | `pytest tests/unit/test_wayfinding_manager.py::test_nav_arrived_verbal -x` | ❌ Wave 0 |
| WAY-05 | NAV_FAILED event triggers verbal explanation and DISPLAY_NAV_STATUS update | unit | `pytest tests/unit/test_wayfinding_manager.py::test_nav_failed_verbal -x` | ❌ Wave 0 |
| DISP-01 | ConnectionManager.send_map_image() sends 0x06 binary frame with PNG payload | unit | `pytest tests/unit/test_display_dispatch.py::test_send_map_image_frame -x` | ❌ Wave 0 |
| DISP-01 | DISPLAY_MAP event triggers map image send via ConnectionManager | unit | `pytest tests/unit/test_display_dispatch.py::test_display_map_event_dispatch -x` | ❌ Wave 0 |
| DISP-02 | ConnectionManager.send_nav_status() sends correct JSON nav_status message | unit | `pytest tests/unit/test_display_dispatch.py::test_send_nav_status_json -x` | ❌ Wave 0 |
| DISP-02 | DISPLAY_NAV_STATUS event triggers nav status send via ConnectionManager | unit | `pytest tests/unit/test_display_dispatch.py::test_display_nav_status_event_dispatch -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `source venv/bin/activate && python -m pytest tests/unit/test_wayfinding_manager.py tests/unit/test_display_dispatch.py -v --tb=short`
- **Per wave merge:** `source venv/bin/activate && python -m pytest tests/unit/ -v --cov=smait --cov-report=term-missing`
- **Phase gate:** Full suite green (80%+ coverage on navigation/ and dialogue/ modules) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_wayfinding_manager.py` — covers WAY-01 through WAY-05; needs MockPOIKnowledgeBase, MockNavController, MockMapManager fixtures
- [ ] `tests/unit/test_display_dispatch.py` — covers DISP-01, DISP-02; needs MockConnectionManager that captures sent frames
- [ ] Test fixture: synthetic 100x100 grey PNG (reuse pattern from `test_map_manager.py`)
- [ ] Test fixture: mock OpenAI response with `tool_calls` (use `unittest.mock.MagicMock` for `response.choices[0].message.tool_calls`)

*(No new framework install needed — pytest + pytest-asyncio are already installed)*

---

## Sources

### Primary (HIGH confidence)
- `smait/dialogue/manager.py` — existing DialogueManager structure; `_ask_api()` extension point verified
- `smait/navigation/map_manager.py` — `render_map()`, `world_to_pixel()`, PIL draw pattern verified
- `smait/navigation/nav_controller.py` — `navigate_to()`, event subscription pattern verified
- `smait/navigation/poi_knowledge_base.py` — `resolve()`, `list_locations()` interface verified
- `smait/core/events.py` — complete EventType enum verified
- `smait/core/config.py` — `NavigationConfig` dataclass pattern verified
- `smait/connection/manager.py` — `send_binary()`, `send_text()`, event subscription pattern verified
- `smait/connection/protocol.py` — `FrameType` IntEnum, `BinaryFrame.pack()` pattern verified
- openai 2.21.0 (venv) — `tools=` parameter syntax verified via installed package version
- Pillow 11.3.0 (venv) — `ImageDraw.ellipse()`, `Image.copy()` API verified

### Secondary (MEDIUM confidence)
- Phase 10 RESEARCH.md (project planning docs) — architectural decisions for navigation layer confirmed
- OpenAI function calling docs (current): Tool call message format (`role: "tool"`, `tool_call_id`) confirmed by installed 2.x API version

### Tertiary (LOW confidence)
- Ollama function calling compatibility — NOT verified for `phi-4-mini`/`mistral-nemo`; marked as unreliable, fallback to OpenAI only

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified in venv; no new deps needed
- Architecture: HIGH — all interfaces (NavController, POIKnowledgeBase, MapManager, DialogueManager) verified by reading source
- OpenAI tool calling: HIGH — package verified at 2.21.0; `tools=` and `tool_calls` response format stable since openai 1.x
- Pitfalls: MEDIUM — tool-call two-round-trip pitfall documented from API knowledge; Ollama unreliability is known project issue (STATE.md)
- Display dispatch: HIGH — binary frame pattern and ConnectionManager extension points verified

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable — openai API and PIL versions locked in venv)
