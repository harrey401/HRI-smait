"""Tests for ChassisClient — CHAS-01 through CHAS-06.

This test suite is the RED phase of TDD. ChassisClient does not exist yet;
tests will fail with ImportError until Plan 02 implements it.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from smait.core.config import Config
from smait.core.events import EventBus, EventType

# ChassisClient does not exist yet — import deferred so pytest can collect tests.
# Each test that needs ChassisClient will get an ImportError at runtime (RED phase).
try:
    from smait.connection.chassis_client import ChassisClient
except ImportError:
    ChassisClient = None  # type: ignore[assignment,misc]


def _require_chassis_client():
    """Raise ImportError if ChassisClient is not yet implemented (RED phase)."""
    if ChassisClient is None:
        raise ImportError(
            "smait.connection.chassis_client.ChassisClient is not implemented yet. "
            "This is the expected RED state — implement ChassisClient in Plan 02."
        )


# ---------------------------------------------------------------------------
# MockChassisServer
# ---------------------------------------------------------------------------

class MockChassisServer:
    """Real WebSocket server for testing ChassisClient."""

    def __init__(self) -> None:
        self.received: list[dict] = []
        self._ws = None
        self._connected_event = asyncio.Event()

    async def handler(self, ws) -> None:
        """WebSocket connection handler — store connection and iterate messages."""
        self._ws = ws
        self._connected_event.set()
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    msg = raw
                self.received.append(msg)
        except Exception:
            pass
        finally:
            self._ws = None
            self._connected_event.clear()

    async def push(self, msg: dict) -> None:
        """Send a JSON message to the connected client."""
        if self._ws is None:
            raise RuntimeError("No client connected to MockChassisServer")
        await self._ws.send(json.dumps(msg))

    async def push_fragments(self, msg: dict) -> None:
        """Send a message split as two consecutive frames (fragment reassembly test)."""
        text = json.dumps(msg)
        half = len(text) // 2
        part1 = text[:half]
        part2 = text[half:]
        # websockets sends as a single message per call; simulate fragment-style
        # by sending two partial messages that the client must buffer-join
        await self._ws.send(part1)
        await self._ws.send(part2)

    async def close(self) -> None:
        """Close the current WebSocket connection (for reconnect testing)."""
        if self._ws is not None:
            await self._ws.close()

    async def wait_connected(self, timeout: float = 5.0) -> None:
        """Block until a client connects (or timeout)."""
        await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def mock_chassis() -> AsyncGenerator[tuple[MockChassisServer, int], None]:
    """Start MockChassisServer on a random OS-assigned port, yield (mock, port)."""
    from websockets.asyncio.server import serve

    mock = MockChassisServer()
    async with serve(mock.handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield mock, port


@pytest_asyncio.fixture
async def chassis_client(mock_chassis: tuple[MockChassisServer, int]) -> AsyncGenerator[ChassisClient, None]:
    """Create and start a ChassisClient pointed at MockChassisServer."""
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    # Accelerate reconnect for tests
    config.chassis.reconnect_max_wait_s = 1.0
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)
    await client.start()
    yield client
    await client.stop()


# ---------------------------------------------------------------------------
# Helper: capture an EventBus emission as an asyncio.Event + payload
# ---------------------------------------------------------------------------

def _capture(event_bus: EventBus, event_type: EventType) -> tuple[asyncio.Event, list]:
    """Subscribe to event_type; return (signal_event, [data]) for assertions."""
    signal = asyncio.Event()
    captured: list = []

    def _handler(data):
        captured.append(data)
        signal.set()

    event_bus.subscribe(event_type, _handler)
    return signal, captured


def _get_event_bus(client: ChassisClient) -> EventBus:
    """Access the EventBus stored on the client (attribute name: event_bus)."""
    return client.event_bus


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connects_to_mock_server(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """CHAS-01: ChassisClient connects to the chassis WS and emits CHASSIS_CONNECTED."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    signal, _ = _capture(event_bus, EventType.CHASSIS_CONNECTED)
    await client.start()
    try:
        await asyncio.wait_for(signal.wait(), timeout=2.0)
    finally:
        await client.stop()

    assert signal.is_set(), "CHASSIS_CONNECTED event was not emitted within 2 s"


@pytest.mark.asyncio
async def test_pose_subscription(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """CHAS-02: Client subscribes to pose topic; mock push triggers CHASSIS_POSE_UPDATE."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    connected, _ = _capture(event_bus, EventType.CHASSIS_CONNECTED)
    pose_signal, pose_data = _capture(event_bus, EventType.CHASSIS_POSE_UPDATE)

    await client.start()
    try:
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        await mock.push({
            "op": "publish",
            "topic": config.chassis.pose_topic,
            "msg": {"x": 1.5, "y": 2.3, "theta": 0.7},
        })
        await asyncio.wait_for(pose_signal.wait(), timeout=2.0)
    finally:
        await client.stop()

    assert pose_signal.is_set()
    data = pose_data[0]
    assert data["x"] == pytest.approx(1.5)
    assert data["y"] == pytest.approx(2.3)
    assert data["theta"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_nav_status_events(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """CHAS-03: Nav status messages with known codes get correct text labels."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    connected, _ = _capture(event_bus, EventType.CHASSIS_CONNECTED)
    nav_signal, nav_data = _capture(event_bus, EventType.CHASSIS_NAV_STATUS)

    await client.start()
    try:
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        await mock.push({
            "op": "publish",
            "topic": config.chassis.nav_status_topic,
            "msg": {"status": 3, "goal_id": "goal-1"},
        })
        await asyncio.wait_for(nav_signal.wait(), timeout=2.0)
    finally:
        await client.stop()

    assert nav_signal.is_set()
    data = nav_data[0]
    assert data["status"] == 3
    assert data["text"] == "succeeded"
    assert data["goal_id"] == "goal-1"


@pytest.mark.asyncio
async def test_robot_state_subscription(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """CHAS-04: Robot status messages get emitted as CHASSIS_STATE_UPDATE."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    connected, _ = _capture(event_bus, EventType.CHASSIS_CONNECTED)
    state_signal, state_data = _capture(event_bus, EventType.CHASSIS_STATE_UPDATE)

    await client.start()
    try:
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        await mock.push({
            "op": "publish",
            "topic": config.chassis.status_topic,
            "msg": {
                "battery": 85.0,
                "nav_status": 600,
                "control_state": 0,
                "velocity": [0.1, 0.0, 0.0],
            },
        })
        await asyncio.wait_for(state_signal.wait(), timeout=2.0)
    finally:
        await client.stop()

    assert state_signal.is_set()
    data = state_data[0]
    assert data["battery"] == pytest.approx(85.0)
    assert data["nav_status"] == 600
    assert data["control_state"] == 0
    assert data["velocity"] == [0.1, 0.0, 0.0]


@pytest.mark.asyncio
async def test_soft_estop(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """CHAS-05: send_soft_stop(True) sends advertise + publish ops to chassis."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    connected, _ = _capture(event_bus, EventType.CHASSIS_CONNECTED)

    await client.start()
    try:
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        await client.send_soft_stop(True)
        # Give a moment for the messages to arrive at the mock
        await asyncio.sleep(0.2)
    finally:
        await client.stop()

    topics = [msg.get("topic") for msg in mock.received]
    ops = [msg.get("op") for msg in mock.received]

    assert config.chassis.soft_stop_topic in topics, (
        f"soft_stop_topic not seen in mock.received; got topics: {topics}"
    )
    assert "advertise" in ops, f"Expected 'advertise' op; got ops: {ops}"
    assert "publish" in ops, f"Expected 'publish' op; got ops: {ops}"

    # Find the publish message for soft_stop and check data
    publish_msgs = [
        m for m in mock.received
        if m.get("op") == "publish" and m.get("topic") == config.chassis.soft_stop_topic
    ]
    assert publish_msgs, "No publish message for soft_stop topic"
    assert publish_msgs[0]["msg"]["data"] is True


@pytest.mark.asyncio
async def test_reconnect_on_disconnect(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """CHAS-06: Client reconnects after server closes the connection and re-subscribes."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    config.chassis.reconnect_max_wait_s = 1.0
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    disconnected_signal = asyncio.Event()
    reconnected_signal = asyncio.Event()
    connect_count = [0]

    def _on_connected(_):
        connect_count[0] += 1
        if connect_count[0] >= 2:
            reconnected_signal.set()

    def _on_disconnected(_):
        disconnected_signal.set()

    event_bus.subscribe(EventType.CHASSIS_CONNECTED, _on_connected)
    event_bus.subscribe(EventType.CHASSIS_DISCONNECTED, _on_disconnected)

    await client.start()
    try:
        # Wait for initial connect
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, lambda: None),
            timeout=0.01,
        )
        # Actually wait for first CONNECTED event
        first_connected = asyncio.Event()
        def _first(_): first_connected.set()
        event_bus.subscribe(EventType.CHASSIS_CONNECTED, _first)

        await asyncio.wait_for(first_connected.wait(), timeout=2.0)

        # Drop the connection — should trigger DISCONNECTED then reconnect
        await mock.close()

        await asyncio.wait_for(disconnected_signal.wait(), timeout=3.0)
        await asyncio.wait_for(reconnected_signal.wait(), timeout=5.0)
    finally:
        await client.stop()

    assert disconnected_signal.is_set(), "CHASSIS_DISCONNECTED never fired"
    assert reconnected_signal.is_set(), "Client did not reconnect (CHASSIS_CONNECTED count < 2)"

    # Verify re-subscription: client should have re-sent subscribe ops after reconnect
    subscribe_ops = [m for m in mock.received if m.get("op") == "subscribe"]
    subscribed_topics = {m.get("topic") for m in subscribe_ops}
    expected_topics = {
        config.chassis.pose_topic,
        config.chassis.status_topic,
        config.chassis.nav_status_topic,
        config.chassis.obstacle_topic,
    }
    assert expected_topics.issubset(subscribed_topics), (
        f"After reconnect, missing subscribe ops. Got: {subscribed_topics}"
    )


@pytest.mark.asyncio
async def test_fragment_reassembly(mock_chassis: tuple[MockChassisServer, int]) -> None:
    """Client reassembles fragmented messages split across multiple WS frames."""
    _require_chassis_client()
    mock, port = mock_chassis
    config = Config()
    config.chassis.host = "localhost"
    config.chassis.port = port
    event_bus = EventBus()
    client = ChassisClient(config, event_bus)

    connected, _ = _capture(event_bus, EventType.CHASSIS_CONNECTED)
    pose_signal, pose_data = _capture(event_bus, EventType.CHASSIS_POSE_UPDATE)

    await client.start()
    try:
        await asyncio.wait_for(connected.wait(), timeout=2.0)
        # Send two half-JSON strings that together form a valid pose publish message
        await mock.push_fragments({
            "op": "publish",
            "topic": config.chassis.pose_topic,
            "msg": {"x": 9.9, "y": 8.8, "theta": 1.1},
        })
        await asyncio.wait_for(pose_signal.wait(), timeout=2.0)
    finally:
        await client.stop()

    assert pose_signal.is_set(), "CHASSIS_POSE_UPDATE not fired after fragment reassembly"
    data = pose_data[0]
    assert data["x"] == pytest.approx(9.9)
    assert data["y"] == pytest.approx(8.8)
    assert data["theta"] == pytest.approx(1.1)
