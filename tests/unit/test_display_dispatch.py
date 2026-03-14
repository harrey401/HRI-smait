"""Unit tests for display dispatch: DISP-01 (map image) and DISP-02 (nav status).

Tests verify:
- FrameType.MAP_IMAGE == 0x06
- MessageSchema.nav_status() produces correct JSON
- ConnectionManager.send_map_image sends a 0x06-prefixed binary frame
- ConnectionManager.send_nav_status sends correct JSON text
- DISPLAY_MAP event triggers send_map_image via EventBus subscription
- DISPLAY_NAV_STATUS event triggers send_nav_status via EventBus subscription
"""

import json
import pytest
from unittest.mock import AsyncMock

from smait.connection.manager import ConnectionManager
from smait.connection.protocol import FrameType, MessageSchema
from smait.core.events import EventBus, EventType


# ---------------------------------------------------------------------------
# Protocol layer tests (no ConnectionManager needed)
# ---------------------------------------------------------------------------


def test_frame_type_map_image():
    """FrameType.MAP_IMAGE must equal 0x06."""
    assert FrameType.MAP_IMAGE == 0x06


def test_message_schema_nav_status():
    """MessageSchema.nav_status returns valid JSON with required keys."""
    raw = MessageSchema.nav_status("navigating", "eng192")
    parsed = json.loads(raw)
    assert parsed["type"] == "nav_status"
    assert parsed["status"] == "navigating"
    assert parsed["destination"] == "eng192"


# ---------------------------------------------------------------------------
# ConnectionManager send method tests
# ---------------------------------------------------------------------------


class TestSendMapImage:
    """Tests for ConnectionManager.send_map_image."""

    @pytest.mark.asyncio
    async def test_send_map_image_frame(self, config, event_bus):
        """send_map_image sends a 0x06-prefixed binary frame containing the PNG payload."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        png_bytes = b"\x89PNG\r\n\x1a\nfake_png_data"
        await manager.send_map_image(png_bytes)

        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        assert isinstance(sent_data, bytes)
        assert sent_data[0] == 0x06
        assert sent_data[1:] == png_bytes


class TestSendNavStatus:
    """Tests for ConnectionManager.send_nav_status."""

    @pytest.mark.asyncio
    async def test_send_nav_status_json(self, config, event_bus):
        """send_nav_status sends a JSON text message with correct fields."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        await manager.send_nav_status("navigating", "eng192")

        mock_ws.send.assert_called_once()
        sent_text = mock_ws.send.call_args[0][0]
        assert isinstance(sent_text, str)
        parsed = json.loads(sent_text)
        assert parsed["type"] == "nav_status"
        assert parsed["status"] == "navigating"
        assert parsed["destination"] == "eng192"


# ---------------------------------------------------------------------------
# EventBus subscription / dispatch tests
# ---------------------------------------------------------------------------


class TestDisplayMapEventDispatch:
    """Tests for DISPLAY_MAP event triggering send_map_image."""

    @pytest.mark.asyncio
    async def test_display_map_event_dispatch(self, config, event_bus):
        """DISPLAY_MAP event causes send_map_image to fire with the PNG payload."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        png_bytes = b"fakepng"
        await event_bus.emit_async(
            EventType.DISPLAY_MAP,
            {"png": png_bytes, "highlighted_poi": "eng192"},
        )

        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        assert isinstance(sent_data, bytes)
        assert sent_data[0] == 0x06
        assert sent_data[1:] == png_bytes


class TestDisplayNavStatusEventDispatch:
    """Tests for DISPLAY_NAV_STATUS event triggering send_nav_status."""

    @pytest.mark.asyncio
    async def test_display_nav_status_event_dispatch(self, config, event_bus):
        """DISPLAY_NAV_STATUS event causes send_nav_status to fire with correct JSON."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        await event_bus.emit_async(
            EventType.DISPLAY_NAV_STATUS,
            {"status": "arrived", "destination": "eng192"},
        )

        mock_ws.send.assert_called_once()
        sent_text = mock_ws.send.call_args[0][0]
        assert isinstance(sent_text, str)
        parsed = json.loads(sent_text)
        assert parsed["type"] == "nav_status"
        assert parsed["status"] == "arrived"
        assert parsed["destination"] == "eng192"
