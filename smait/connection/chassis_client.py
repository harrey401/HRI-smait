"""ChassisClient — outbound WebSocket client for the chassis rosbridge-inspired protocol.

Connects to the chassis at configurable host:port, subscribes to 4 topics,
translates received messages to SMAIT EventBus events, and provides send_soft_stop
and call_service utilities.

Auto-reconnects on disconnect using the websockets 16 `async for ws in connect(uri)` pattern.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import Any

from websockets.asyncio.client import connect

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Navigation status code → human-readable text
# ---------------------------------------------------------------------------
NAV_STATUS_MAP: dict[int, str] = {
    0: "pending",
    1: "active",
    2: "preempted",
    3: "succeeded",
    4: "aborted",
}


class ChassisClient:
    """WebSocket client that speaks the chassis rosbridge-inspired JSON protocol.

    Usage::

        client = ChassisClient(config, event_bus)
        await client.start()          # begins background _run() task
        ...
        await client.stop()           # graceful shutdown
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._cfg = config.chassis
        self._nav_cfg = config.navigation
        self._bus = event_bus
        self._running: bool = False
        self._connected: asyncio.Event = asyncio.Event()
        self._id_gen = itertools.count(1)
        self._pending_calls: dict[str, asyncio.Future] = {}
        # Rosbridge fragment reassembly: keyed by fragment id
        self._fragments: dict[str, list] = {}
        # Buffer for raw partial-JSON frames that aren't yet valid JSON
        self._json_buffer: str = ""
        self._ws = None
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True if the WebSocket connection is currently established."""
        return self._connected.is_set()

    @property
    def event_bus(self) -> EventBus:
        """The EventBus this client emits to (used by tests)."""
        return self._bus

    def _next_id(self) -> str:
        return f"smait-{next(self._id_gen)}"

    async def start(self) -> None:
        """Begin the background connection loop."""
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._run())

    async def stop(self) -> None:
        """Gracefully shut down the client."""
        self._running = False
        self._connected.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main connection loop — uses websockets async-for-reconnect pattern."""
        uri = f"ws://{self._cfg.host}:{self._cfg.port}"
        try:
            async for ws in connect(uri):
                self._json_buffer = ""
                try:
                    self._ws = ws
                    self._connected.set()
                    self._bus.emit(EventType.CHASSIS_CONNECTED)
                    logger.info("ChassisClient connected to %s", uri)

                    await self._setup_subscriptions(ws)

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            self._process_raw(raw)
                        except Exception:
                            logger.exception("Error processing chassis message: %r", raw)

                except Exception:
                    logger.exception("ChassisClient connection error")
                finally:
                    self._connected.clear()
                    self._ws = None
                    # Cancel all pending call_service futures
                    for fut in list(self._pending_calls.values()):
                        if not fut.done():
                            fut.cancel()
                    self._pending_calls.clear()
                    self._bus.emit(EventType.CHASSIS_DISCONNECTED)
                    logger.info("ChassisClient disconnected from %s", uri)

                if not self._running:
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("ChassisClient _run exited unexpectedly")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _setup_subscriptions(self, ws) -> None:
        """Send subscribe ops for all chassis topics."""
        subscriptions = [
            (self._cfg.pose_topic, "geometry_msgs/Pose2D"),
            (self._cfg.status_topic, "yutong_assistance/RobotStatus"),
            (self._cfg.nav_status_topic, "actionlib_msgs/GoalStatus"),
            (self._cfg.obstacle_topic, "std_msgs/Int8"),
            (self._nav_cfg.path_topic, "yutong_assistance/point_array"),
        ]
        for topic, msg_type in subscriptions:
            op = {
                "op": "subscribe",
                "id": self._next_id(),
                "topic": topic,
                "type": msg_type,
            }
            await ws.send(json.dumps(op))

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _process_raw(self, raw: str) -> None:
        """Try to parse raw as JSON; buffer partial frames and retry on accumulation."""
        # Try combined buffer + new raw first
        combined = self._json_buffer + raw
        try:
            msg = json.loads(combined)
            self._json_buffer = ""
            self._handle_message(msg)
        except json.JSONDecodeError:
            # Not valid JSON yet — buffer for next frame
            self._json_buffer = combined

    def _handle_message(self, msg: dict) -> None:
        """Route a fully-parsed message to the correct handler."""
        op = msg.get("op")
        if op == "fragment":
            self._handle_fragment(msg)
        elif op == "publish":
            self._handle_publish(msg)
        elif op == "service_response":
            self._handle_service_response(msg)
        elif op == "png":
            self._handle_png(msg)
        else:
            logger.debug("ChassisClient: unhandled op=%r", op)

    def _handle_publish(self, msg: dict) -> None:
        """Translate a publish message to the appropriate EventBus event."""
        topic = msg.get("topic", "")
        data = msg.get("msg", {})

        if topic == self._cfg.pose_topic:
            self._bus.emit(EventType.CHASSIS_POSE_UPDATE, {
                "x": data.get("x", 0.0),
                "y": data.get("y", 0.0),
                "theta": data.get("theta", 0.0),
            })

        elif topic == self._cfg.nav_status_topic:
            status = data.get("status", 0)
            self._bus.emit(EventType.CHASSIS_NAV_STATUS, {
                "status": status,
                "text": NAV_STATUS_MAP.get(status, "unknown"),
                "goal_id": data.get("goal_id", ""),
            })

        elif topic == self._cfg.status_topic:
            self._bus.emit(EventType.CHASSIS_STATE_UPDATE, {
                "battery": data.get("battery", 0.0),
                "nav_status": data.get("nav_status", 0),
                "control_state": data.get("control_state", 0),
                "velocity": data.get("velocity", []),
            })

        elif topic == self._cfg.obstacle_topic:
            self._bus.emit(EventType.CHASSIS_OBSTACLE, {
                "region": data.get("data", 0),
            })

        elif topic == self._nav_cfg.path_topic:
            points = list(zip(data.get("px", []), data.get("py", [])))
            self._bus.emit(EventType.CHASSIS_PATH_UPDATE, {"points": points})

        else:
            logger.debug("ChassisClient: unhandled publish topic=%r", topic)

    def _handle_fragment(self, msg: dict) -> None:
        """Reassemble rosbridge-style fragmented messages.

        Fragment fields: id, num (0-based index), total, data (string chunk).
        When all parts arrive, join and re-process as a complete message.
        """
        frag_id = msg.get("id")
        total = msg.get("total", 1)
        num = msg.get("num", 0)
        chunk = msg.get("data", "")

        if frag_id not in self._fragments:
            self._fragments[frag_id] = [None] * total

        parts = self._fragments[frag_id]
        if 0 <= num < len(parts):
            parts[num] = chunk

        if all(p is not None for p in parts):
            assembled = "".join(parts)
            del self._fragments[frag_id]
            try:
                assembled_msg = json.loads(assembled)
                self._handle_message(assembled_msg)
            except json.JSONDecodeError:
                logger.error("ChassisClient: failed to parse assembled fragment id=%r", frag_id)

    def _handle_service_response(self, msg: dict) -> None:
        """Resolve a pending call_service Future with the response."""
        call_id = msg.get("id")
        if call_id and call_id in self._pending_calls:
            fut = self._pending_calls.pop(call_id)
            if not fut.done():
                fut.set_result(msg.get("values", {}))

    # ------------------------------------------------------------------
    # Outgoing commands
    # ------------------------------------------------------------------

    async def send_soft_stop(self, stop: bool = True) -> None:
        """Send soft e-stop command to the chassis.

        Sends an advertise op followed by a publish op for the soft_stop_topic.

        Args:
            stop: True to activate soft stop, False to release.

        Raises:
            RuntimeError: If not currently connected to the chassis.
        """
        if not self._connected.is_set() or self._ws is None:
            raise RuntimeError("ChassisClient: cannot send_soft_stop — not connected")

        topic = self._cfg.soft_stop_topic

        advertise_op = {
            "op": "advertise",
            "id": self._next_id(),
            "topic": topic,
            "type": "std_msgs/Bool",
        }
        await self._ws.send(json.dumps(advertise_op))

        publish_op = {
            "op": "publish",
            "id": self._next_id(),
            "topic": topic,
            "msg": {"data": stop},
        }
        await self._ws.send(json.dumps(publish_op))

    async def call_service(
        self,
        service: str,
        args: dict | None = None,
        timeout: float = 5.0,
    ) -> dict:
        """Call a ROS service via the chassis rosbridge.

        Args:
            service: ROS service name (e.g., "/move_base").
            args: Service request arguments dict. Defaults to {}.
            timeout: Seconds to wait for response before raising TimeoutError.

        Returns:
            The service response values dict.

        Raises:
            RuntimeError: If not connected.
            asyncio.TimeoutError: If no response arrives within timeout seconds.
        """
        if not self._connected.is_set() or self._ws is None:
            raise RuntimeError("ChassisClient: cannot call_service — not connected")

        call_id = self._next_id()
        op = {
            "op": "call_service",
            "id": call_id,
            "service": service,
            "args": args or {},
        }

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_calls[call_id] = fut

        await self._ws.send(json.dumps(op))

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_calls.pop(call_id, None)
            raise

    def _handle_png(self, msg: dict) -> None:
        """Handle op:png messages — emit CHASSIS_MAP_UPDATE with raw msg dict."""
        self._bus.emit(EventType.CHASSIS_MAP_UPDATE, msg)

    async def subscribe_topic(
        self,
        topic: str,
        msg_type: str,
        **kwargs: Any,
    ) -> None:
        """Send a subscribe op for a topic, with optional extra fields.

        Unlike _setup_subscriptions, this is called manually by MapManager
        (e.g., for the map topic with compression/fragment_size options).

        Args:
            topic: ROS topic name.
            msg_type: ROS message type string.
            **kwargs: Optional extra fields (e.g., compression, fragment_size).

        Raises:
            RuntimeError: If not currently connected to the chassis.
        """
        if not self._connected.is_set() or self._ws is None:
            raise RuntimeError("ChassisClient: cannot subscribe_topic — not connected")

        op: dict[str, Any] = {
            "op": "subscribe",
            "id": self._next_id(),
            "topic": topic,
            "type": msg_type,
        }
        op.update(kwargs)
        await self._ws.send(json.dumps(op))

    async def send_cancel_navigation(self) -> None:
        """Send a cancel navigation command to the chassis.

        Sends an advertise op followed by a publish op for the cancel_nav_topic.

        Raises:
            RuntimeError: If not currently connected to the chassis.
        """
        if not self._connected.is_set() or self._ws is None:
            raise RuntimeError("ChassisClient: cannot send_cancel_navigation — not connected")

        topic = self._nav_cfg.cancel_nav_topic

        advertise_op = {
            "op": "advertise",
            "id": self._next_id(),
            "topic": topic,
            "type": "actionlib_msgs/GoalID",
        }
        await self._ws.send(json.dumps(advertise_op))

        publish_op = {
            "op": "publish",
            "id": self._next_id(),
            "topic": topic,
            "msg": {"stamp": "", "id": ""},
        }
        await self._ws.send(json.dumps(publish_op))

    async def send_insert_marker(self, name: str) -> None:
        """Insert a named POI marker at the current robot pose.

        Sends an advertise op followed by a publish op for the insert_marker_topic.

        Args:
            name: POI marker name to insert.

        Raises:
            RuntimeError: If not currently connected to the chassis.
        """
        if not self._connected.is_set() or self._ws is None:
            raise RuntimeError("ChassisClient: cannot send_insert_marker — not connected")

        topic = self._nav_cfg.insert_marker_topic

        advertise_op = {
            "op": "advertise",
            "id": self._next_id(),
            "topic": topic,
            "type": "yutong_assistance/poi_msgs",
        }
        await self._ws.send(json.dumps(advertise_op))

        publish_op = {
            "op": "publish",
            "id": self._next_id(),
            "topic": topic,
            "msg": {
                "name": name,
                "behavior_code": 0,
                "time_out": 0,
                "rest_time": 0,
            },
        }
        await self._ws.send(json.dumps(publish_op))
