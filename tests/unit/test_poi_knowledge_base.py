"""Tests for POIKnowledgeBase — POI-01 through POI-04, SETUP-02.

RED phase tests — all tests fail with NotImplementedError until Plan 03 implements POIKnowledgeBase.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from smait.core.config import Config
from smait.core.events import EventBus
from smait.navigation.poi_knowledge_base import POIKnowledgeBase


def make_chassis(config: Config, event_bus: EventBus) -> MagicMock:
    """Create a mock ChassisClient."""
    chassis = MagicMock()
    chassis.call_service = AsyncMock()
    chassis.send_insert_marker = AsyncMock()
    chassis.event_bus = event_bus
    return chassis


@pytest.mark.asyncio
async def test_fetch_markers():
    """POI-01: fetch_markers() calls /marker_operation/get_markers and returns waypoint list."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {
        "waypoints": [
            {"name": "eng192", "x": 1.0, "y": 2.0},
            {"name": "restroom_1", "x": 3.0, "y": 4.0},
        ]
    }
    kb = POIKnowledgeBase(config, bus, chassis)

    markers = await kb.fetch_markers()

    assert isinstance(markers, list)
    assert len(markers) >= 1
    chassis.call_service.assert_called_once_with("/marker_operation/get_markers")


@pytest.mark.asyncio
async def test_add_marker():
    """POI-02: add_marker() calls chassis.send_insert_marker with the marker name."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    kb = POIKnowledgeBase(config, bus, chassis)

    await kb.add_marker("test_poi")

    chassis.send_insert_marker.assert_called_once_with("test_poi")


@pytest.mark.asyncio
async def test_delete_marker():
    """POI-03: delete_marker() calls /marker_manager/delete_poi with the correct name."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    chassis.call_service.return_value = {"success": True}
    kb = POIKnowledgeBase(config, bus, chassis)

    result = await kb.delete_marker("old_poi")

    assert result is True
    chassis.call_service.assert_called_once()
    call_args = str(chassis.call_service.call_args)
    assert "/marker_manager/delete_poi" in call_args
    assert "old_poi" in call_args


def test_resolve_human_name():
    """POI-04: resolve() performs case-insensitive lookup and returns chassis ID or None."""
    config = Config()
    bus = EventBus()
    chassis = make_chassis(config, bus)
    kb = POIKnowledgeBase(config, bus, chassis)

    # Inject mappings directly (bypassing load for skeleton test)
    kb._mappings = {
        "room eng192": "eng192",
        "bathroom": "restroom_1",
    }

    # Case-insensitive match: "Room ENG192" → "eng192"
    result = kb.resolve("Room ENG192")
    assert result == "eng192", f"Expected 'eng192', got {result!r}"

    # Exact match lowercase
    result = kb.resolve("bathroom")
    assert result == "restroom_1"

    # Non-existent entry → None
    result = kb.resolve("nonexistent")
    assert result is None


def test_load_json_config(tmp_path: Path):
    """SETUP-02: load() reads a JSON config file and populates _mappings."""
    # Create temp POI config
    tefa_dir = tmp_path / "tefa"
    tefa_dir.mkdir()
    poi_file = tefa_dir / "3.json"
    poi_data = {
        "room ENG192": "eng192",
        "bathroom": "restroom_1",
    }
    poi_file.write_text(json.dumps(poi_data))

    config = Config()
    config.navigation.poi_config_dir = str(tmp_path)
    bus = EventBus()
    chassis = make_chassis(config, bus)
    kb = POIKnowledgeBase(config, bus, chassis)

    kb.load("tefa", "3")

    # After load, _mappings should have normalized (lowercase) keys
    assert kb._mappings.get("room eng192") == "eng192" or kb._mappings.get("room ENG192") == "eng192"
    assert "restroom_1" in kb._mappings.values()
