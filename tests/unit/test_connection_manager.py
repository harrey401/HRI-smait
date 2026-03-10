"""Unit tests for ConnectionManager TTS audio forwarding via 0x05 frames."""

import pytest
from unittest.mock import AsyncMock, patch

from smait.connection.manager import ConnectionManager
from smait.core.events import EventBus, EventType


class TestConnectionManagerTTSForwarding:
    """Tests for ConnectionManager forwarding TTS_AUDIO_CHUNK events to WebSocket."""

    @pytest.mark.asyncio
    async def test_tts_audio_forwarded(self, config, event_bus):
        """TTS_AUDIO_CHUNK with dict payload sends 0x05-prefixed frame to WebSocket."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        pcm_bytes = b'\x00' * 100
        await event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm_bytes})

        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        assert sent_data[0] == 0x05
        assert sent_data[1:] == pcm_bytes

    @pytest.mark.asyncio
    async def test_tts_audio_raw_bytes_forwarded(self, config, event_bus):
        """TTS_AUDIO_CHUNK with raw bytes payload sends 0x05-prefixed frame to WebSocket."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        raw_pcm = b'\xFF\xFE' * 50  # 100 bytes
        await event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, raw_pcm)

        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        assert sent_data[0] == 0x05
        assert sent_data[1:] == raw_pcm

    @pytest.mark.asyncio
    async def test_tts_audio_no_client(self, config, event_bus):
        """When no client is connected, send_tts_audio does not raise."""
        manager = ConnectionManager(config, event_bus)
        # _client remains None (default — no connected WebSocket)

        pcm_bytes = b'\xAB\xCD' * 10
        # Should not raise any exception
        await event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"audio": pcm_bytes})

    @pytest.mark.asyncio
    async def test_tts_audio_dict_missing_audio_key(self, config, event_bus):
        """TTS_AUDIO_CHUNK dict without 'audio' key does not send to WebSocket."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        # Dict payload without 'audio' key — _on_tts_audio_chunk should not forward it
        await event_bus.emit_async(EventType.TTS_AUDIO_CHUNK, {"text": "hello"})

        mock_ws.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_tts_audio_directly(self, config, event_bus):
        """send_tts_audio() directly packs and sends the 0x05 frame."""
        manager = ConnectionManager(config, event_bus)
        mock_ws = AsyncMock()
        manager._client = mock_ws

        pcm = b'\x01\x02\x03\x04'
        await manager.send_tts_audio(pcm)

        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        assert sent_data == bytes([0x05]) + pcm

    @pytest.mark.asyncio
    async def test_send_tts_audio_no_client_no_raise(self, config, event_bus):
        """send_tts_audio() with no connected client does not raise."""
        manager = ConnectionManager(config, event_bus)
        # No client connected
        await manager.send_tts_audio(b'\x00' * 50)  # Should not raise
