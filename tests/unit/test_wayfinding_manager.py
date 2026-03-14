"""Tests for WayfindingManager — WAY-01 through WAY-05.

RED phase tests (Task 1 of Plan 01) fail with NotImplementedError until Task 2 implements handlers.
Task 2 adds test_render_map_highlight_pixel (GREEN).
Plan 03 Task 1 adds 5 new tests for DialogueManager tool-call loop and
WayfindingManager NAV_ARRIVED/NAV_FAILED verbal handlers (RED).
Plan 03 Task 2 implements those tests to GREEN.
"""

from __future__ import annotations

import base64
import io
import json
import struct

import pytest
import pytest_asyncio
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch, call

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.navigation.wayfinding_manager import WayfindingManager, WAYFINDING_TOOLS
from smait.navigation.map_manager import MapManager
from smait.dialogue.manager import DialogueManager, DialogueResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_grey_png(width: int = 100, height: int = 100) -> bytes:
    """Create a synthetic grey PNG image as bytes."""
    img = Image.new("RGBA", (width, height), color=(200, 200, 200, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_event_bus():
    """EventBus instance for capturing emitted events."""
    return EventBus()


@pytest.fixture
def mock_poi_kb():
    """Mock POIKnowledgeBase."""
    kb = MagicMock()
    kb.resolve = MagicMock(return_value="eng192")
    kb.list_locations = MagicMock(return_value=["eng192", "eng101", "library", "cafe", "gym"])
    return kb


@pytest.fixture
def mock_nav_controller():
    """Mock NavController with async navigate_to."""
    nav = MagicMock()
    nav.navigate_to = AsyncMock(return_value={"success": True})
    return nav


@pytest.fixture
def mock_map_manager():
    """Mock MapManager with render_map_with_highlight."""
    mm = MagicMock()
    mm.render_map_with_highlight = MagicMock(return_value=make_grey_png())
    return mm


@pytest.fixture
def wayfinding_manager(mock_event_bus, mock_poi_kb, mock_nav_controller, mock_map_manager):
    """WayfindingManager with all mocks and default Config."""
    config = Config()
    return WayfindingManager(
        config=config,
        event_bus=mock_event_bus,
        poi_kb=mock_poi_kb,
        nav_controller=mock_nav_controller,
        map_manager=mock_map_manager,
    )


# ---------------------------------------------------------------------------
# WAY-01: Tool registration
# ---------------------------------------------------------------------------


def test_tool_registration(wayfinding_manager):
    """WAY-01: get_tools() returns list with 2 tool dicts named query_location and navigate_to."""
    tools = wayfinding_manager.get_tools()
    assert isinstance(tools, list)
    assert len(tools) == 2
    names = [t["function"]["name"] for t in tools]
    assert "query_location" in names
    assert "navigate_to" in names


def test_tool_handlers(wayfinding_manager):
    """WAY-01: get_tool_handlers() returns dict with query_location and navigate_to async callables."""
    handlers = wayfinding_manager.get_tool_handlers()
    assert isinstance(handlers, dict)
    assert "query_location" in handlers
    assert "navigate_to" in handlers
    import asyncio
    assert asyncio.iscoroutinefunction(handlers["query_location"])
    assert asyncio.iscoroutinefunction(handlers["navigate_to"])


# ---------------------------------------------------------------------------
# WAY-01: query_location handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_location_found(wayfinding_manager, mock_poi_kb):
    """WAY-01: query_location calls resolve and returns found=True with poi_name."""
    result = await wayfinding_manager._handle_query_location({"location_name": "room ENG192"})

    mock_poi_kb.resolve.assert_called_once_with("room ENG192")
    assert result["found"] is True
    assert result["poi_name"] == "eng192"
    assert "verbal" in result


@pytest.mark.asyncio
async def test_query_location_not_found(wayfinding_manager, mock_poi_kb):
    """WAY-01: query_location returns found=False with 'I don't know' verbal when resolve returns None."""
    mock_poi_kb.resolve.return_value = None

    result = await wayfinding_manager._handle_query_location({"location_name": "narnia"})

    assert result["found"] is False
    assert "I don't know" in result["verbal"]


# ---------------------------------------------------------------------------
# WAY-03: query_location dispatches DISPLAY_MAP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_location_dispatches_map(wayfinding_manager, mock_event_bus, mock_map_manager):
    """WAY-03: query_location emits DISPLAY_MAP with png bytes and highlighted_poi."""
    captured_events: list[dict] = []

    def capture(data):
        captured_events.append(data)

    mock_event_bus.subscribe(EventType.DISPLAY_MAP, capture)

    await wayfinding_manager._handle_query_location({"location_name": "eng192"})

    assert len(captured_events) == 1
    event_data = captured_events[0]
    assert isinstance(event_data["png"], bytes)
    assert event_data["highlighted_poi"] == "eng192"


# ---------------------------------------------------------------------------
# WAY-02: navigate_to handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigate_to_success(wayfinding_manager, mock_nav_controller):
    """WAY-02: navigate_to returns started=True and verbal confirmation on success."""
    mock_nav_controller.navigate_to.return_value = {"success": True}

    result = await wayfinding_manager._handle_navigate_to({"poi_name": "eng192"})

    mock_nav_controller.navigate_to.assert_called_once_with("eng192")
    assert result["started"] is True
    assert "verbal" in result


@pytest.mark.asyncio
async def test_navigate_to_failure(wayfinding_manager, mock_nav_controller):
    """WAY-02: navigate_to returns started=False with failure verbal when nav fails."""
    mock_nav_controller.navigate_to.return_value = {"success": False}

    result = await wayfinding_manager._handle_navigate_to({"poi_name": "eng192"})

    assert result["started"] is False
    assert "verbal" in result


# ---------------------------------------------------------------------------
# WAY-04: navigate_to dispatches DISPLAY_NAV_STATUS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigate_to_dispatches_status(wayfinding_manager, mock_event_bus, mock_nav_controller):
    """WAY-04: navigate_to emits DISPLAY_NAV_STATUS with status=navigating on success."""
    mock_nav_controller.navigate_to.return_value = {"success": True}
    captured_events: list[dict] = []

    def capture(data):
        captured_events.append(data)

    mock_event_bus.subscribe(EventType.DISPLAY_NAV_STATUS, capture)

    await wayfinding_manager._handle_navigate_to({"poi_name": "eng192"})

    assert len(captured_events) == 1
    event_data = captured_events[0]
    assert event_data["status"] == "navigating"
    assert event_data["destination"] == "eng192"


# ---------------------------------------------------------------------------
# WAY-03: MapManager.render_map_with_highlight draws circle at POI pixel
# ---------------------------------------------------------------------------


def test_render_map_highlight_pixel():
    """WAY-03: render_map_with_highlight draws a non-grey pixel at the POI location.

    Uses a 100x100 grey synthetic map with unit metadata so that world coords
    map 1:1 to pixels. POI at world (50, 50) should land at pixel (50, 50)
    in the image and be rendered with a yellow highlight circle.
    """
    # Create a synthetic 100x100 grey map image
    config = Config()
    bus = EventBus()
    chassis_mock = MagicMock()
    chassis_mock.subscribe_topic = AsyncMock()
    chassis_mock.call_service = AsyncMock()
    manager = MapManager(config, bus, chassis_mock)

    # Load a synthetic grey map directly
    grey_img = Image.new("RGBA", (100, 100), color=(200, 200, 200, 255))
    manager._map_image = grey_img
    manager._map_meta = {
        "origin_x": 0.0,
        "origin_y": 0.0,
        "resolution": 1.0,
        "width": 100,
        "height": 100,
    }

    # Place POI at world coords (50, 50) — should map to pixel (50, 50)
    # With origin=(0,0), resolution=1.0, height=100:
    #   col = (50 - 0) / 1.0 = 50
    #   row = 100 - (50 - 0) / 1.0 = 50
    manager._poi_positions["eng192"] = {"x": 50.0, "y": 50.0}

    png_bytes = manager.render_map_with_highlight("eng192")

    assert isinstance(png_bytes, bytes)
    assert png_bytes[:4] == b"\x89PNG"

    # Decode the PNG and check the pixel at (50, 50) is not grey
    result_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    pixel = result_img.getpixel((50, 50))
    grey_pixel = (200, 200, 200, 255)
    assert pixel != grey_pixel, (
        f"Expected a non-grey pixel at (50, 50) indicating highlight circle was drawn, "
        f"got {pixel}"
    )


# ---------------------------------------------------------------------------
# Plan 03: DialogueManager tool registration and tool-call loop (WAY-05)
# ---------------------------------------------------------------------------


@pytest.fixture
def dialogue_config():
    """Minimal Config for DialogueManager."""
    return Config()


@pytest.fixture
def dialogue_event_bus():
    """EventBus for DialogueManager tests."""
    return EventBus()


@pytest.fixture
def dialogue_manager(dialogue_config, dialogue_event_bus):
    """DialogueManager with mock OpenAI client."""
    dm = DialogueManager(config=dialogue_config, event_bus=dialogue_event_bus)
    return dm


def test_dialogue_register_tools(dialogue_manager):
    """WAY-05: register_tools() sets _tools and _tool_handlers on DialogueManager."""
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    handlers = {"test_tool": AsyncMock()}

    dialogue_manager.register_tools(tools, handlers)

    assert dialogue_manager._tools == tools
    assert dialogue_manager._tool_handlers == handlers


@pytest.mark.asyncio
async def test_dialogue_tool_call_flow(dialogue_manager):
    """WAY-05: When OpenAI returns tool_calls, DialogueManager executes the handler,
    injects the tool result as a 'tool' role message, makes a second LLM call,
    and returns the second response text as DialogueResponse.
    """
    # Set up a mock handler that returns a dict result
    mock_handler = AsyncMock(return_value={"found": True, "poi_name": "eng192"})
    tools = [{"type": "function", "function": {"name": "query_location"}}]
    dialogue_manager.register_tools(tools, {"query_location": mock_handler})

    # Build first response (has tool_calls)
    first_tool_call = MagicMock()
    first_tool_call.id = "call_123"
    first_tool_call.type = "function"
    first_tool_call.function.name = "query_location"
    first_tool_call.function.arguments = json.dumps({"location_name": "ENG192"})
    first_tool_call.model_dump.return_value = {
        "id": "call_123", "type": "function",
        "function": {"name": "query_location", "arguments": '{"location_name":"ENG192"}'}
    }

    first_message = MagicMock()
    first_message.tool_calls = [first_tool_call]
    first_message.content = None

    first_choice = MagicMock()
    first_choice.message = first_message
    first_response = MagicMock()
    first_response.choices = [first_choice]
    first_response.usage = MagicMock()
    first_response.usage.total_tokens = 50

    # Build second response (text, no tool_calls)
    second_message = MagicMock()
    second_message.content = "ENG192 is on the third floor!"
    second_message.tool_calls = None

    second_choice = MagicMock()
    second_choice.message = second_message
    second_response = MagicMock()
    second_response.choices = [second_choice]
    second_response.usage = MagicMock()
    second_response.usage.total_tokens = 30

    # Wire mock OpenAI client with side_effect for 2 calls
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[first_response, second_response]
    )
    dialogue_manager._openai_client = mock_client

    result = await dialogue_manager.ask("where is ENG192?")

    # Handler must be called with correct parsed args
    mock_handler.assert_called_once_with({"location_name": "ENG192"})

    # OpenAI must be called twice (first: with tools; second: follow-up without tools)
    assert mock_client.chat.completions.create.call_count == 2

    # Returned response must contain the second-call text
    assert isinstance(result, DialogueResponse)
    assert result.text == "ENG192 is on the third floor!"


@pytest.mark.asyncio
async def test_dialogue_no_tools_passthrough(dialogue_manager):
    """WAY-05: When no tools registered, _ask_api behaves identically to original
    (no tools= param sent to OpenAI).
    """
    # No register_tools() called — _tools should default to []

    # Build a plain text response
    plain_message = MagicMock()
    plain_message.content = "Hello there!"
    plain_message.tool_calls = None

    plain_choice = MagicMock()
    plain_choice.message = plain_message
    plain_response = MagicMock()
    plain_response.choices = [plain_choice]
    plain_response.usage = MagicMock()
    plain_response.usage.total_tokens = 10

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=plain_response)
    dialogue_manager._openai_client = mock_client

    # Disable Ollama so we go straight to API
    dialogue_manager._config = MagicMock()
    dialogue_manager._config.try_local_first = False
    dialogue_manager._config.api_model = "gpt-4o-mini"
    dialogue_manager._config.max_tokens = 256
    dialogue_manager._config.temperature = 0.7
    dialogue_manager._config.system_prompt = "You are Jackie."
    dialogue_manager._config.max_history_turns = 10

    result = await dialogue_manager.ask("hi")

    # tools= must NOT be in the call kwargs
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" not in call_kwargs, "tools= should not be sent when _tools is empty"
    assert result.text == "Hello there!"


# ---------------------------------------------------------------------------
# Plan 03: WayfindingManager NAV_ARRIVED / NAV_FAILED verbal handlers (WAY-05)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nav_arrived_verbal(wayfinding_manager, mock_event_bus):
    """WAY-05: NAV_ARRIVED event triggers DIALOGUE_RESPONSE with 'arrived' text
    and DISPLAY_NAV_STATUS with status=arrived.
    """
    dialogue_events: list = []
    display_events: list = []

    mock_event_bus.subscribe(EventType.DIALOGUE_RESPONSE, dialogue_events.append)
    mock_event_bus.subscribe(EventType.DISPLAY_NAV_STATUS, display_events.append)

    wayfinding_manager._on_nav_arrived({"destination": "ENG192"})

    # DISPLAY_NAV_STATUS should fire
    assert len(display_events) == 1
    assert display_events[0]["status"] == "arrived"
    assert display_events[0]["destination"] == "ENG192"

    # DIALOGUE_RESPONSE should fire with a DialogueResponse
    assert len(dialogue_events) == 1
    response = dialogue_events[0]
    assert isinstance(response, DialogueResponse)
    assert "arrived" in response.text.lower()
    assert "ENG192" in response.text


@pytest.mark.asyncio
async def test_nav_failed_verbal(wayfinding_manager, mock_event_bus):
    """WAY-05: NAV_FAILED event triggers DIALOGUE_RESPONSE with 'wasn't able to reach' text
    and DISPLAY_NAV_STATUS with status=failed.
    """
    dialogue_events: list = []
    display_events: list = []

    mock_event_bus.subscribe(EventType.DIALOGUE_RESPONSE, dialogue_events.append)
    mock_event_bus.subscribe(EventType.DISPLAY_NAV_STATUS, display_events.append)

    wayfinding_manager._on_nav_failed({"destination": "ENG192", "reason": "obstacle"})

    # DISPLAY_NAV_STATUS should fire
    assert len(display_events) == 1
    assert display_events[0]["status"] == "failed"
    assert display_events[0]["destination"] == "ENG192"

    # DIALOGUE_RESPONSE should fire with 'wasn't able to reach' phrasing
    assert len(dialogue_events) == 1
    response = dialogue_events[0]
    assert isinstance(response, DialogueResponse)
    assert "wasn't able to reach" in response.text
    assert "ENG192" in response.text
