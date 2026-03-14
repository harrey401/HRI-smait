"""Tests for WayfindingManager — WAY-01 through WAY-04.

RED phase tests (Task 1) fail with NotImplementedError until Task 2 implements handlers.
Task 2 adds test_render_map_highlight_pixel (GREEN).
"""

from __future__ import annotations

import base64
import io

import pytest
import pytest_asyncio
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, call

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.navigation.wayfinding_manager import WayfindingManager, WAYFINDING_TOOLS


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
