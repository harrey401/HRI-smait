"""WebSocket server — connection handling and frame demuxing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import websockets
from websockets.asyncio.server import Server, ServerConnection

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.connection.protocol import BinaryFrame, FrameType, MessageSchema

logger = logging.getLogger(__name__)


class ConnectionManager:
    """WebSocket server that accepts Jackie's connection.

    - Listens on 0.0.0.0:8765
    - Single client connection (one robot)
    - Demuxes incoming binary frames by type byte
    - Handles incoming JSON messages (DOA, TTS state, config, CAE status)
    - Provides send methods for outbound data
    - Heartbeat: ping every 5s, timeout after 15s
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.connection
        self._event_bus = event_bus
        self._server: Optional[Server] = None
        self._client: Optional[ServerConnection] = None
        self._client_connected = asyncio.Event()
        self._last_pong: float = 0.0
        self._running = False

        # Subscribe to events we need to forward to Jackie
        event_bus.subscribe(EventType.TTS_AUDIO_CHUNK, self._on_tts_audio_chunk)
        event_bus.subscribe(EventType.TTS_START, self._on_tts_start)
        event_bus.subscribe(EventType.TTS_END, self._on_tts_end)

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self._config.host,
            self._config.port,
            ping_interval=self._config.heartbeat_interval_s,
            ping_timeout=self._config.heartbeat_interval_s * 3,
            max_size=2**22,  # 4MB max frame size
        )
        logger.info("WebSocket server listening on %s:%d",
                     self._config.host, self._config.port)

    async def stop(self) -> None:
        """Stop the server and disconnect client."""
        self._running = False
        if self._client:
            await self._client.close()
            self._client = None
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("WebSocket server stopped")

    async def wait_for_connection(self, timeout: float = 0) -> bool:
        """Wait until a client connects. Returns True if connected."""
        if timeout > 0:
            try:
                await asyncio.wait_for(self._client_connected.wait(), timeout)
            except asyncio.TimeoutError:
                return False
        else:
            await self._client_connected.wait()
        return True

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a single Jackie connection."""
        if self._client is not None:
            logger.warning("Rejecting second client connection")
            await websocket.close(1008, "Only one client allowed")
            return

        self._client = websocket
        self._last_pong = time.monotonic()
        self._client_connected.set()
        logger.info("Jackie connected from %s", websocket.remote_address)
        self._event_bus.emit(EventType.CONNECTION_OPEN)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    self._handle_binary(message)
                elif isinstance(message, str):
                    self._handle_text(message)
        except websockets.ConnectionClosed as e:
            logger.info("Jackie disconnected: %s", e)
        except Exception:
            logger.exception("Error in client handler")
        finally:
            self._client = None
            self._client_connected.clear()
            self._event_bus.emit(EventType.CONNECTION_CLOSED)
            logger.info("Jackie connection closed")

    def _handle_binary(self, data: bytes) -> None:
        """Demux incoming binary frames by type byte."""
        if len(data) < 2:
            return

        try:
            frame = BinaryFrame.from_bytes(data)
        except ValueError:
            logger.warning("Unknown binary frame type: 0x%02x", data[0])
            return

        if frame.frame_type == FrameType.AUDIO_CAE:
            self._event_bus.emit(EventType.SPEECH_DETECTED, {
                "audio": frame.payload,
                "type": "cae",
                "timestamp": time.monotonic(),
            })
        elif frame.frame_type == FrameType.VIDEO:
            self._event_bus.emit(EventType.FACE_UPDATED, {
                "jpeg": frame.payload,
                "type": "video",
                "timestamp": time.monotonic(),
            })
        elif frame.frame_type == FrameType.AUDIO_RAW:
            self._event_bus.emit(EventType.SPEECH_DETECTED, {
                "audio": frame.payload,
                "type": "raw",
                "timestamp": time.monotonic(),
            })

    def _handle_text(self, raw: str) -> None:
        """Handle incoming JSON text frames from Jackie."""
        try:
            msg = MessageSchema.parse_text_message(raw)
        except Exception:
            logger.warning("Invalid JSON from Jackie: %s", raw[:200])
            return

        msg_type = msg.get("type")
        if msg_type == "doa":
            self._event_bus.emit(EventType.DOA_UPDATE, {
                "angle": msg.get("angle", 0),
                "beam": msg.get("beam", 0),
            })
        elif msg_type == "tts_state":
            # Jackie reporting its local TTS state (if using Android TTS fallback)
            pass
        elif msg_type == "config":
            logger.info("Received config update from Jackie: %s", msg)
        elif msg_type == "cae_status":
            self._event_bus.emit(EventType.CAE_STATUS, {
                "aec": msg.get("aec", False),
                "beamforming": msg.get("beamforming", False),
                "noise_suppression": msg.get("noise_suppression", False),
            })
        else:
            logger.debug("Unhandled text message type: %s", msg_type)

    # --- Outbound send methods ---

    async def send_text(self, message: str) -> None:
        """Send a JSON text frame to Jackie."""
        if self._client:
            try:
                await self._client.send(message)
            except Exception:
                logger.warning("Failed to send text to Jackie")

    async def send_binary(self, data: bytes) -> None:
        """Send a binary frame to Jackie."""
        if self._client:
            try:
                await self._client.send(data)
            except Exception:
                logger.warning("Failed to send binary to Jackie")

    async def send_tts_audio(self, pcm_bytes: bytes) -> None:
        """Send TTS audio to Jackie as a 0x05 binary frame."""
        frame = BinaryFrame.pack(FrameType.TTS_AUDIO, pcm_bytes)
        await self.send_binary(frame)

    async def send_state(self, state: str, robot_status: str) -> None:
        """Send state update to Jackie UI."""
        await self.send_text(MessageSchema.state(state, robot_status))

    async def send_transcript(self, text: str, speaker: str) -> None:
        """Send transcript to Jackie chat display."""
        await self.send_text(MessageSchema.transcript(text, speaker))

    async def send_response(self, text: str) -> None:
        """Send full response text to Jackie."""
        await self.send_text(MessageSchema.response(text))

    async def send_tts_control(self, action: str) -> None:
        """Send TTS mic gating control to Jackie."""
        await self.send_text(MessageSchema.tts_control(action))

    # --- Event handlers ---

    async def _on_tts_audio_chunk(self, data: Any) -> None:
        """Forward TTS audio chunks to Jackie."""
        if isinstance(data, dict) and "audio" in data:
            await self.send_tts_audio(data["audio"])
        elif isinstance(data, bytes):
            await self.send_tts_audio(data)

    async def _on_tts_start(self, _data: Any) -> None:
        await self.send_tts_control("start")

    async def _on_tts_end(self, _data: Any) -> None:
        await self.send_tts_control("end")
