"""Tests for MapManager — MAP-01 through MAP-04, NAV-02, SETUP-03.

RED phase tests — all tests fail with NotImplementedError until Plan 02 implements MapManager.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
import pytest_asyncio
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.navigation.map_manager import MapManager, world_to_pixel, draw_robot_arrow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_map_png_b64(width: int = 10, height: int = 10) -> str:
    """Generate a synthetic PNG image and return it as a base64-encoded string."""
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_chassis(config: Config, event_bus: EventBus) -> MagicMock:
    """Create a mock ChassisClient with async call_service."""
    chassis = MagicMock()
    chassis.call_service = AsyncMock()
    chassis.subscribe_topic = AsyncMock()
    chassis.event_bus = event_bus
    return chassis


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_map_png_decode():
    """MAP-01: MapManager decodes a base64 PNG from op:png message correctly."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    png_b64 = make_test_map_png_b64(10, 10)
    msg = {
        "op": "png",
        "data": png_b64,
        "width": 10,
        "height": 10,
        "origin_x": -5.0,
        "origin_y": -5.0,
        "resolution": 0.1,
    }

    # Call manager to process the PNG message — requires implementation
    await manager.start()  # NotImplementedError expected


@pytest.mark.asyncio
async def test_list_maps():
    """MAP-02: list_maps() calls /get_map_info service and returns building list."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {
        "list_info": [
            {
                "building_name": "tefa",
                "floor_info": [{"floor_name": "3"}],
            }
        ]
    }
    manager = MapManager(config, bus, chassis)

    result = await manager.list_maps()

    building_names = [b.get("building_name") for b in result]
    assert "tefa" in building_names


@pytest.mark.asyncio
async def test_switch_map():
    """MAP-03: switch_map() calls /node_manager_control with cmd=7 for the target floor."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {"result": 0}
    manager = MapManager(config, bus, chassis)

    result = await manager.switch_map("tefa", "3")

    assert result is True
    chassis.call_service.assert_called_once()
    call_args = chassis.call_service.call_args
    # First arg is service name
    assert "/node_manager_control" in str(call_args)
    # args should include cmd=7
    assert "7" in str(call_args) or 7 in str(call_args).split()


def test_robot_arrow_position():
    """MAP-04: world_to_pixel maps (0,0) to center pixel for a 100x100 map centered at origin."""
    meta = {
        "origin_x": -5.0,
        "origin_y": -5.0,
        "resolution": 0.1,
        "height": 100,
        "width": 100,
    }
    # world (0, 0) should map to pixel (50, 50) — center of a 100x100 map
    # with origin at (-5, -5) and resolution 0.1 m/px
    px, py = world_to_pixel(0.0, 0.0, meta)
    assert px == 50
    assert py == 50


def test_render_map_returns_png_bytes():
    """MAP-04 continued: render_map() returns valid PNG bytes when map is loaded."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    png_bytes = manager.render_map()

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # Verify it's valid PNG (PNG magic bytes: \x89PNG)
    assert png_bytes[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_path_overlay():
    """NAV-02: Path points are rendered as an overlay on the map PNG."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    # Provide path points and verify render output
    manager._path_points = [(0.0, 0.0), (1.0, 1.0)]

    png_bytes = manager.render_map()

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0


@pytest.mark.asyncio
async def test_startup_auto_detect():
    """SETUP-03: On CHASSIS_CONNECTED, MapManager queries active map and emits MAP_ACTIVE_FLOOR."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)

    # Mock responses for auto-detection services
    async def mock_call_service(service, args=None, timeout=5.0):
        if service == "/get_map_info":
            return {
                "list_info": [
                    {
                        "building_name": "tefa",
                        "floor_info": [{"floor_name": "3", "current": True}],
                    }
                ]
            }
        elif service == "/layered_map_cmd":
            return {"result": 0}
        return {}

    chassis.call_service = AsyncMock(side_effect=mock_call_service)

    emitted_events = []

    def capture_event(data):
        emitted_events.append(data)

    bus.subscribe(EventType.MAP_ACTIVE_FLOOR, capture_event)

    manager = MapManager(config, bus, chassis)
    await manager.start()

    # Simulate CHASSIS_CONNECTED event to trigger auto-detect
    bus.emit(EventType.CHASSIS_CONNECTED)

    # Allow async tasks to run
    import asyncio
    await asyncio.sleep(0.05)

    assert any(
        e and e.get("building") == "tefa" for e in emitted_events
    ), f"Expected MAP_ACTIVE_FLOOR event, got: {emitted_events}"


def test_on_map_update_emits_rendered_png():
    """MAP-01: _on_map_update decodes flat-format PNG and emits MAP_RENDERED with valid PNG bytes."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    # Subscribe handlers directly (bypasses start() which calls async subscribe_topic)
    bus.subscribe(EventType.CHASSIS_MAP_UPDATE, manager._on_map_update)

    rendered_events: list = []

    def capture(data):
        rendered_events.append(data)

    bus.subscribe(EventType.MAP_RENDERED, capture)

    png_b64 = make_test_map_png_b64(10, 10)
    bus.emit(
        EventType.CHASSIS_MAP_UPDATE,
        {
            "data": png_b64,
            "origin_x": -5.0,
            "origin_y": -5.0,
            "resolution": 0.05,
            "width": 10,
            "height": 10,
        },
    )

    assert len(rendered_events) == 1
    png_out = rendered_events[0]
    assert isinstance(png_out, bytes)
    assert png_out[:4] == b"\x89PNG"
    assert manager._map_image is not None


def test_on_map_update_rosbridge_nested_format():
    """MAP-01: _on_map_update handles nested rosbridge op:png format and sets correct metadata."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    # Subscribe handlers directly
    bus.subscribe(EventType.CHASSIS_MAP_UPDATE, manager._on_map_update)

    rendered_events: list = []

    def capture(data):
        rendered_events.append(data)

    bus.subscribe(EventType.MAP_RENDERED, capture)

    png_b64 = make_test_map_png_b64(10, 10)
    bus.emit(
        EventType.CHASSIS_MAP_UPDATE,
        {
            "msg": {
                "data": png_b64,
                "info": {
                    "origin": {"position": {"x": -5.0, "y": -5.0}},
                    "resolution": 0.05,
                    "width": 10,
                    "height": 10,
                },
            }
        },
    )

    assert len(rendered_events) == 1
    png_out = rendered_events[0]
    assert isinstance(png_out, bytes)
    assert png_out[:4] == b"\x89PNG"
    assert manager._map_meta.get("origin_x") == -5.0
    assert manager._map_meta.get("origin_y") == -5.0


def test_on_pose_update_rerenders_when_map_loaded():
    """MAP-04: _on_pose_update re-renders map when map is loaded, skips render when no map."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    # Subscribe handlers directly
    bus.subscribe(EventType.CHASSIS_POSE_UPDATE, manager._on_pose_update)

    rendered_events: list = []

    def capture(data):
        rendered_events.append(data)

    bus.subscribe(EventType.MAP_RENDERED, capture)

    # Pose update WITHOUT a map loaded — should NOT emit MAP_RENDERED
    bus.emit(EventType.CHASSIS_POSE_UPDATE, {"x": 1.0, "y": 2.0, "theta": 0.5})
    assert len(rendered_events) == 0, "Expected no MAP_RENDERED when map is not loaded"

    # Now load a map manually
    png_b64 = make_test_map_png_b64(10, 10)
    img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGBA")
    manager._map_image = img
    manager._map_meta = {
        "origin_x": -5.0,
        "origin_y": -5.0,
        "resolution": 0.05,
        "width": 10,
        "height": 10,
    }

    # Pose update WITH a map loaded — should emit MAP_RENDERED
    bus.emit(EventType.CHASSIS_POSE_UPDATE, {"x": 1.0, "y": 2.0, "theta": 0.5})
    assert len(rendered_events) == 1
    assert manager._current_pose == {"x": 1.0, "y": 2.0, "theta": 0.5}


def test_on_path_update_rerenders_when_map_loaded():
    """MAP-04: _on_path_update re-renders map when map is loaded and stores path points."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    manager = MapManager(config, bus, chassis)

    # Subscribe handlers directly
    bus.subscribe(EventType.CHASSIS_PATH_UPDATE, manager._on_path_update)

    rendered_events: list = []

    def capture(data):
        rendered_events.append(data)

    bus.subscribe(EventType.MAP_RENDERED, capture)

    # Load a map manually
    png_b64 = make_test_map_png_b64(10, 10)
    img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGBA")
    manager._map_image = img
    manager._map_meta = {
        "origin_x": -5.0,
        "origin_y": -5.0,
        "resolution": 0.05,
        "width": 10,
        "height": 10,
    }

    path_points = [(1.0, 1.0), (2.0, 2.0)]
    bus.emit(EventType.CHASSIS_PATH_UPDATE, {"points": path_points})

    assert len(rendered_events) == 1
    assert manager._path_points == path_points
