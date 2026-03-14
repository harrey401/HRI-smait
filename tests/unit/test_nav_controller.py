"""Tests for NavController — NAV-01, NAV-03, NAV-04, NAV-05.

RED phase tests — all tests fail with NotImplementedError until Plan 04 implements NavController.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.navigation.nav_controller import NavController
from smait.navigation.poi_knowledge_base import POIKnowledgeBase


def make_chassis(config: Config, event_bus: EventBus) -> MagicMock:
    """Create a mock ChassisClient."""
    chassis = MagicMock()
    chassis.call_service = AsyncMock()
    chassis.send_cancel_navigation = AsyncMock()
    chassis.event_bus = event_bus
    return chassis


def make_poi_kb(config: Config, event_bus: EventBus, chassis) -> MagicMock:
    """Create a mock POIKnowledgeBase."""
    poi_kb = MagicMock(spec=POIKnowledgeBase)
    poi_kb.resolve = MagicMock(return_value="eng192")
    return poi_kb


@pytest.mark.asyncio
async def test_navigate_to_poi():
    """NAV-01: navigate_to() calls the chassis /poi service with the POI name."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {
        "success": True,
        "avaliable_list": [],
    }
    poi_kb = make_poi_kb(config, bus, chassis)
    controller = NavController(config, bus, chassis, poi_kb)

    result = await controller.navigate_to("eng192")

    chassis.call_service.assert_called_once()
    call_args = str(chassis.call_service.call_args)
    assert "/poi" in call_args
    assert "eng192" in call_args
    assert result.get("success") is True


@pytest.mark.asyncio
async def test_navigate_to_poi_not_found():
    """NAV-02: navigate_to() emits NAV_FAILED with reason='poi_not_found' when /poi returns success=False."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {
        "success": False,
        "avaliable_list": ["eng192", "restroom_1"],
    }
    poi_kb = make_poi_kb(config, bus, chassis)
    controller = NavController(config, bus, chassis, poi_kb)

    failed_events = []
    bus.subscribe(EventType.NAV_FAILED, failed_events.append)

    result = await controller.navigate_to("nonexistent")

    assert len(failed_events) == 1, f"Expected NAV_FAILED, got {failed_events}"
    event_data = failed_events[0]
    assert event_data["reason"] == "poi_not_found"
    assert event_data["destination"] == "nonexistent"
    assert "eng192" in event_data["available"]
    assert result.get("success") is False


@pytest.mark.asyncio
async def test_nav_status_events():
    """NAV-03: NAV_ARRIVED is emitted for status=3 (succeeded), NAV_FAILED for status=4 (aborted)."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    poi_kb = make_poi_kb(config, bus, chassis)
    controller = NavController(config, bus, chassis, poi_kb)

    arrived_events = []
    failed_events = []

    bus.subscribe(EventType.NAV_ARRIVED, arrived_events.append)
    bus.subscribe(EventType.NAV_FAILED, failed_events.append)

    # Simulate navigation start so controller tracks destination
    controller._navigating = True

    # Simulate CHASSIS_NAV_STATUS with status=3 (succeeded)
    bus.emit(EventType.CHASSIS_NAV_STATUS, {"status": 3, "text": "succeeded", "goal_id": "g1"})
    await asyncio.sleep(0.05)

    assert len(arrived_events) == 1, f"Expected NAV_ARRIVED, got {arrived_events}"

    # Reset and simulate status=4 (aborted)
    controller._navigating = True
    bus.emit(EventType.CHASSIS_NAV_STATUS, {"status": 4, "text": "aborted", "goal_id": "g2"})
    await asyncio.sleep(0.05)

    assert len(failed_events) == 1, f"Expected NAV_FAILED, got {failed_events}"


@pytest.mark.asyncio
async def test_cancel_navigation():
    """NAV-04: cancel_navigation() calls chassis.send_cancel_navigation."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    poi_kb = make_poi_kb(config, bus, chassis)
    controller = NavController(config, bus, chassis, poi_kb)

    await controller.cancel_navigation()

    chassis.send_cancel_navigation.assert_called_once()


@pytest.mark.asyncio
async def test_calculate_distance():
    """NAV-05: calculate_distance() calls /calculate_distance service and returns the distance."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {"distance": 42.5}
    poi_kb = make_poi_kb(config, bus, chassis)
    controller = NavController(config, bus, chassis, poi_kb)

    distance = await controller.calculate_distance(0.0, 0.0, "3", 5.0, 5.0, "3")

    assert distance == 42.5
    chassis.call_service.assert_called_once()
    call_args = str(chassis.call_service.call_args)
    assert "/calculate_distance" in call_args


@pytest.mark.asyncio
async def test_startup_auto_detect():
    """SETUP-03: CHASSIS_CONNECTED triggers floor detection + POI config load."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)

    # /get_map_info returns building + floor info
    chassis.call_service.return_value = {
        "building_name": "tefa",
        "floor_name": "3",
    }

    poi_kb = make_poi_kb(config, bus, chassis)
    poi_kb.load = MagicMock()
    poi_kb.fetch_markers = AsyncMock(return_value=[])

    floor_events = []
    bus.subscribe(EventType.MAP_ACTIVE_FLOOR, floor_events.append)

    controller = NavController(config, bus, chassis, poi_kb)  # noqa: F841

    # Fire CHASSIS_CONNECTED — should trigger on_chassis_connected
    bus.emit(EventType.CHASSIS_CONNECTED, None)

    # Allow async handler to run
    await asyncio.sleep(0.05)

    poi_kb.load.assert_called_once_with("tefa", "3")
    poi_kb.fetch_markers.assert_called_once()
    assert len(floor_events) == 1
    assert floor_events[0] == {"building": "tefa", "floor": "3"}
