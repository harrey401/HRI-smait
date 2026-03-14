# Phase 9: Chassis WebSocket Client — Research

**Researched:** 2026-03-13
**Domain:** Python asyncio WebSocket client — rosbridge-style JSON protocol, in-process mock server testing
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CHAS-01 | SMAIT server connects to chassis WebSocket at configurable IP/port | `websockets.asyncio.client.connect` as async-for iterator handles reconnect; config via new `ChassisConfig` dataclass |
| CHAS-02 | Server subscribes to robot pose (x, y, theta) at regular intervals | Subscribe op → `/robot_pose` topic; `asyncio.create_task` for subscription management; emit `CHASSIS_POSE_UPDATE` EventBus event |
| CHAS-03 | Server subscribes to navigation status (running/success/failed/cancelled) | Subscribe to `/navi_status`; translate status codes 0-4 to EventBus events `CHASSIS_NAV_STATUS` |
| CHAS-04 | Server retrieves robot global state (battery, nav_status, velocity, control_state) | Subscribe to `/robot_status`; emit `CHASSIS_STATE_UPDATE` EventBus event |
| CHAS-05 | Server can send soft e-stop command to chassis | Advertise `/soft_stop`, publish `{"data": true}` — uses publish pattern |
| CHAS-06 | Connection auto-reconnects on disconnect with exponential backoff | `async for ws in connect(uri)` — built-in reconnect with backoff in websockets 16.0; also custom backoff cap |
</phase_requirements>

---

## Summary

Phase 9 adds a new outbound WebSocket client (`ChassisClient`) that speaks the chassis's rosbridge-inspired JSON protocol. The SMAIT codebase already has an inbound WebSocket server (`ConnectionManager`) using `websockets` 16.0; the chassis client follows the same library but uses the `websockets.asyncio.client.connect` API.

The chassis protocol uses three operation patterns: **subscribe** (client tells server what topics to push), **publish** (client sends commands via advertise/publish/unadvertise), and **call_service** (request-response). These are well-defined in the additional context and do not require hardware access to implement — everything can be tested against an in-process mock server.

The key testing insight is that `websockets.asyncio.server.serve(handler, 'localhost', 0)` binds on port 0 (OS assigns free port), giving a full real WebSocket server inside `pytest-asyncio` tests without any subprocess or network mocking. This is the same approach used in the websockets library's own test suite.

**Primary recommendation:** Implement `ChassisClient` as a self-contained asyncio class in `smait/connection/chassis_client.py`, add `ChassisConfig` to `smait/core/config.py`, add 6 new `EventType` members to `smait/core/events.py`, and test everything against an in-process `MockChassisServer` fixture in `tests/unit/test_chassis_client.py`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `websockets` | 16.0 (already installed) | Async WebSocket client + mock server | Already in codebase; `connect` supports both context-manager and infinite-iterator reconnect patterns |
| `pytest-asyncio` | current (already installed) | Run `async def test_*` functions | Already in codebase (see `pyproject.toml` dev deps) |
| `asyncio` | stdlib | Event loop, tasks, queues, timeouts | All existing SMAIT code is asyncio-native |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `dataclasses` | stdlib | `ChassisConfig` config section | Matches existing config pattern (16 dataclasses in `config.py`) |
| `asyncio.Queue` | stdlib | Buffer incoming chassis messages | Decouple receive loop from message processing |
| `asyncio.Event` | stdlib | Signal connection ready / stopping | Same pattern as `_client_connected` in `ConnectionManager` |
| `uuid` / `itertools.count` | stdlib | Generate `id` fields for subscribe/call_service ops | IDs must be unique per session; simple counter is sufficient |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| websockets `async for ws in connect(uri)` reconnect | Manual reconnect loop with `asyncio.sleep` | Built-in iterator is simpler and already tested; manual loop is more controllable if backoff cap is needed (note: built-in uses exponential backoff automatically) |
| In-process `serve(handler, 'localhost', 0)` mock | `unittest.mock.patch` websocket | Real server tests the actual JSON serialization and framing; mock would only test that `.send()` was called |
| `asyncio.Queue` for message routing | Dict of asyncio.Future per `id` | Queue is simpler; Future-per-id is needed only for `call_service` response correlation |

**Installation:** No new packages needed. `websockets` 16.0 is already installed.

---

## Architecture Patterns

### Recommended Project Structure
```
smait/
├── connection/
│   ├── manager.py           # Existing — inbound Android WS server (unchanged)
│   └── chassis_client.py    # NEW — outbound chassis WS client
├── core/
│   ├── config.py            # ADD ChassisConfig dataclass + field in Config
│   └── events.py            # ADD 6 chassis EventType members
tests/
└── unit/
    └── test_chassis_client.py  # NEW — all chassis tests using MockChassisServer
```

### Pattern 1: ChassisClient Class Structure

**What:** Single asyncio class that owns the WebSocket connection lifecycle, subscription state, and message routing to EventBus.

**When to use:** Matches `ConnectionManager` shape; easy to inject in tests.

```python
# smait/connection/chassis_client.py
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


class ChassisClient:
    """Outbound WebSocket client to the chassis controller.

    Speaks the rosbridge-inspired JSON protocol:
      subscribe / publish / call_service
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._cfg = config.chassis
        self._bus = event_bus
        self._running = False
        self._connected = asyncio.Event()
        self._id_gen = itertools.count(1)
        self._pending_calls: dict[str, asyncio.Future] = {}

    def _next_id(self) -> str:
        return f"smait-{next(self._id_gen)}"

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        self._connected.clear()

    async def _run(self) -> None:
        uri = f"ws://{self._cfg.host}:{self._cfg.port}"
        async for ws in connect(uri):
            try:
                self._connected.set()
                logger.info("Connected to chassis at %s", uri)
                await self._setup_subscriptions(ws)
                async for raw in ws:
                    self._handle_message(json.loads(raw))
            except Exception:
                logger.exception("Chassis connection error")
            finally:
                self._connected.clear()
                if not self._running:
                    break

    async def _setup_subscriptions(self, ws) -> None:
        """Send all subscribe ops after connecting."""
        subs = [
            ("/robot_pose",   "geometry_msgs/Pose2D"),
            ("/robot_status", "yutong_assistance/RobotStatus"),
            ("/navi_status",  "actionlib_msgs/GoalStatus"),
            ("/obstacle_region", "std_msgs/Int8"),
        ]
        for topic, msg_type in subs:
            sub_id = self._next_id()
            await ws.send(json.dumps({
                "op": "subscribe",
                "id": sub_id,
                "topic": topic,
                "type": msg_type,
            }))
```

### Pattern 2: In-Process Mock Chassis Server (Test Fixture)

**What:** `pytest-asyncio` fixture that starts a real WebSocket server on a random port. The server mimics the chassis by responding to subscribe/publish/call_service ops.

**When to use:** Every chassis client test. No external process needed.

```python
# tests/unit/test_chassis_client.py
import asyncio
import json
import pytest
from websockets.asyncio.server import serve

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus, EventType


class MockChassisServer:
    """In-process chassis mock. Stores received ops, sends scripted responses."""

    def __init__(self):
        self.received: list[dict] = []
        self.responses: asyncio.Queue = asyncio.Queue()
        self._ws = None

    async def handler(self, ws):
        self._ws = ws
        async for raw in ws:
            msg = json.loads(raw)
            self.received.append(msg)
            # Process queued responses
            while not self.responses.empty():
                resp = await self.responses.get()
                await ws.send(json.dumps(resp))

    async def push(self, msg: dict) -> None:
        """Push a message to the connected client."""
        if self._ws:
            await self._ws.send(json.dumps(msg))


@pytest.fixture
async def mock_chassis():
    """Starts a real WS server on OS-assigned port; yields (server, client_config)."""
    mock = MockChassisServer()
    async with serve(mock.handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield mock, port
```

### Pattern 3: EventType Extensions for Chassis

**What:** Add 6 new EventType members that chassis events map to. Follows existing convention in `events.py`.

```python
# Addition to smait/core/events.py — in the EventType enum

# Chassis / Navigation (new in Phase 9)
CHASSIS_POSE_UPDATE = auto()      # data: {"x": float, "y": float, "theta": float}
CHASSIS_NAV_STATUS = auto()       # data: {"status": int, "text": str, "goal_id": str}
CHASSIS_STATE_UPDATE = auto()     # data: {"battery": float, "nav_status": int,
                                  #        "control_state": int, "velocity": [...]}
CHASSIS_OBSTACLE = auto()         # data: {"region": int}  0=none,1=right,2=ahead,4=left
CHASSIS_CONNECTED = auto()        # data: None
CHASSIS_DISCONNECTED = auto()     # data: None
```

### Pattern 4: ChassisConfig Dataclass

**What:** New config section added to `smait/core/config.py`, following the existing 16-dataclass pattern.

```python
@dataclass
class ChassisConfig:
    host: str = "192.168.20.22"
    port: int = 9090          # Typical rosbridge default; confirm with real chassis
    reconnect_max_wait_s: float = 30.0
    pose_topic: str = "/robot_pose"
    status_topic: str = "/robot_status"
    nav_status_topic: str = "/navi_status"
    obstacle_topic: str = "/obstacle_region"
    soft_stop_topic: str = "/soft_stop"
```

Then add `chassis: ChassisConfig = field(default_factory=ChassisConfig)` to the `Config` dataclass.

### Pattern 5: Fragment Reassembly

**What:** If a chassis message is split into fragments, reassemble before parsing.

**When to use:** Any topic where `msg_type` data might exceed the chassis `fragment_size`. In practice, pose and status messages are small. Map images (Phase 10) will definitely fragment.

```python
# In ChassisClient._handle_message():
def _handle_message(self, msg: dict) -> None:
    op = msg.get("op")
    if op == "fragment":
        self._handle_fragment(msg)
        return
    if op == "publish":
        self._handle_publish(msg)
    elif op == "service_response":
        self._handle_service_response(msg)

_fragments: dict[str, list] = {}

def _handle_fragment(self, msg: dict) -> None:
    frag_id = msg["id"]
    num = msg["num"]       # 0-indexed position
    total = msg["total"]
    data = msg["data"]

    if frag_id not in self._fragments:
        self._fragments[frag_id] = [None] * total
    self._fragments[frag_id][num] = data

    if all(f is not None for f in self._fragments[frag_id]):
        reassembled = json.loads("".join(self._fragments.pop(frag_id)))
        self._handle_message(reassembled)
```

### Anti-Patterns to Avoid
- **Blocking the event loop in the receive loop:** Never use `time.sleep()` or synchronous I/O in `_run()`. Use `asyncio.sleep()` for any delays.
- **Using `asyncio.wait_for` on the entire `_run` task:** The reconnect loop must run indefinitely; wrap only individual `recv()` / `send()` calls when a timeout is needed.
- **Sharing `pending_calls` dict across reconnects without clearing:** On reconnect, all in-flight `call_service` Futures must be cancelled and removed.
- **Hard-coding chassis port 9090:** Make it configurable — the port is only an assumption (rosbridge default). Confirm with real chassis in Phase 13.
- **Not unsubscribing on clean shutdown:** Send `{"op": "unsubscribe"}` before closing to be a good client (defensive; chassis may not require it).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Exponential backoff on reconnect | Custom `asyncio.sleep(2**n)` loop | `async for ws in connect(uri)` (websockets 16.0 built-in) | websockets built-in uses jittered exponential backoff; capped at ~60s by default; already tested |
| In-process test server | `subprocess.Popen`, Docker, mocking | `websockets.asyncio.server.serve(handler, 'localhost', 0)` | Port 0 = OS picks free port; no cleanup needed; tests run offline |
| JSON schema validation | Custom type-checking dicts | `dataclasses` + simple `.get()` with defaults | Protocol is internal; excessive validation overhead not worth it for known chassis messages |
| Message ID generation | UUID library | `itertools.count(1)` wrapped as `f"smait-{next(counter)}"` | Deterministic IDs make tests simpler; uniqueness only needs to hold per session |

**Key insight:** `websockets` 16.0 already solves reconnection with its `async for ws in connect(uri)` pattern. The only custom logic needed is re-sending subscribe ops after each reconnect, which is 5–10 lines.

---

## Common Pitfalls

### Pitfall 1: `asyncio.Event` not reset on reconnect
**What goes wrong:** `self._connected` stays set after disconnect; callers think the client is connected when it isn't.
**Why it happens:** The `finally` block in `_run` clears it, but only if `_run` is structured correctly. If an exception is swallowed before `finally`, the event stays set.
**How to avoid:** Always clear `_connected` in a `finally` block wrapping the per-connection scope, not the outer reconnect loop.
**Warning signs:** Tests for "client emits CHASSIS_DISCONNECTED on mock server close" pass but real disconnect handling breaks.

### Pitfall 2: Subscribe ops sent once — not re-sent on reconnect
**What goes wrong:** After the first connect, subscriptions work. After reconnect, the client receives no data because the chassis server has no memory of the old subscriptions.
**Why it happens:** The subscribe ops are sent inside the first `async with connect(...)` block and are not repeated.
**How to avoid:** Call `_setup_subscriptions(ws)` at the start of every iteration of the `async for ws in connect(uri)` loop.
**Warning signs:** Mock server test simulates disconnect-reconnect; after reconnect, no CHASSIS_POSE_UPDATE events arrive.

### Pitfall 3: `call_service` Future leak on reconnect
**What goes wrong:** If a `call_service` is in-flight and the connection drops, the Future in `_pending_calls` is never resolved. Awaiting it hangs forever.
**Why it happens:** The `service_response` handler never fires because the connection was lost.
**How to avoid:** On disconnect (in the `finally` block), cancel and remove all entries in `_pending_calls` with `CancelledError`.
**Warning signs:** Tests for e-stop that simulate mid-call disconnect hang indefinitely.

### Pitfall 4: websockets 16.0 `serve` context manager vs `start_server`
**What goes wrong:** Code copied from older websockets examples uses `websockets.serve(...)` (legacy sync-style), which behaves differently in 16.0.
**Why it happens:** websockets went through a major API refactor; 16.0 uses `websockets.asyncio.server.serve` (async context manager class).
**How to avoid:** Always import from `websockets.asyncio.server` and `websockets.asyncio.client` — not from the top-level `websockets` module.
**Warning signs:** `DeprecationWarning: remove the handler argument` or `AttributeError: module 'websockets' has no attribute 'asyncio'`.

### Pitfall 5: Fragment reassembly only needed for large payloads
**What goes wrong:** Phase 9 topics (pose, status, navi_status) are all small JSON objects. A fragment handler that doesn't handle the `num=0` case or assumes 1-indexed `num` will silently corrupt map images in Phase 10.
**Why it happens:** Phase 9 code "works" without testing fragment reassembly; Phase 10 breaks on first large map response.
**How to avoid:** Implement fragment reassembly in Phase 9 even though it won't be exercised by pose/status data. Include a unit test with a synthetic fragment sequence.
**Warning signs:** Map PNG in Phase 10 is corrupted or partially rendered.

---

## Code Examples

Verified patterns from official websockets 16.0 (confirmed in venv):

### In-process mock server (port 0 pattern)
```python
# Source: verified in SMAIT venv — websockets 16.0
from websockets.asyncio.server import serve
import asyncio, json

async def test_example():
    received = []

    async def handler(ws):
        async for raw in ws:
            received.append(json.loads(raw))
            await ws.send(json.dumps({"op": "service_response", "result": True}))

    async with serve(handler, "localhost", 0) as server:
        port = server.sockets[0].getsockname()[1]
        # port is OS-assigned, always free
        from websockets.asyncio.client import connect
        async with connect(f"ws://localhost:{port}") as ws:
            await ws.send(json.dumps({"op": "call_service", "id": "1", "service": "/test"}))
            resp = json.loads(await ws.recv())
            assert resp["result"] is True
```

### Reconnect iterator (built-in exponential backoff)
```python
# Source: websockets 16.0 docs — async for iterator reconnects automatically
from websockets.asyncio.client import connect
import asyncio, json

async def _run(uri: str, running_flag):
    async for ws in connect(uri):          # reconnects on ConnectionClosed
        try:
            await _setup_subscriptions(ws)
            async for raw in ws:
                handle(json.loads(raw))
        except Exception:
            pass                           # log, then outer loop reconnects
        finally:
            if not running_flag:
                break
```

### Subscribe op (chassis protocol)
```python
# Source: protocol documented in additional_context
import json

def make_subscribe(topic: str, msg_type: str, sub_id: str) -> str:
    return json.dumps({
        "op": "subscribe",
        "id": sub_id,
        "topic": topic,
        "type": msg_type,
    })

def make_publish(topic: str, pub_id: str, msg: dict) -> str:
    return json.dumps({
        "op": "publish",
        "topic": topic,
        "id": pub_id,
        "msg": msg,
    })

def make_advertise(topic: str, msg_type: str, pub_id: str) -> str:
    return json.dumps({
        "op": "advertise",
        "id": pub_id,
        "topic": topic,
        "type": msg_type,
    })
```

### Soft e-stop publish sequence
```python
# Advertise once on connect; publish on demand
async def send_soft_stop(ws, stop: bool, adv_id: str, pub_id: str) -> None:
    # Step 1: advertise (idempotent — chassis ignores duplicate advertise)
    await ws.send(json.dumps({
        "op": "advertise",
        "id": adv_id,
        "topic": "/soft_stop",
        "type": "std_msgs/Bool",
    }))
    # Step 2: publish
    await ws.send(json.dumps({
        "op": "publish",
        "topic": "/soft_stop",
        "id": pub_id,
        "msg": {"data": stop},
    }))
```

### Navigation status code mapping
```python
# /navi_status status field — actionlib_msgs/GoalStatus codes
NAV_STATUS_MAP = {
    0: "pending",
    1: "active",        # running
    2: "preempted",     # cancelled
    3: "succeeded",     # success
    4: "aborted",       # failed
}

# /robot_status nav_status field — yutong_assistance/RobotStatus codes
ROBOT_NAV_STATUS_MAP = {
    600: "idle",
    601: "navigating",
    602: "succeeded",
    603: "failed",
    604: "cancelled",
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `websockets.connect()` coroutine (must wrap in retry loop) | `async for ws in connect(uri)` iterator (built-in reconnect) | websockets 10.x | Saves ~20 lines of custom backoff code |
| `websockets.serve(handler, host, port)` (legacy) | `websockets.asyncio.server.serve(handler, host, port)` (async ctx mgr) | websockets 14.x | Must import from `websockets.asyncio.server`, not top-level |
| `asyncio.ensure_future` | `asyncio.create_task` | Python 3.7 | `create_task` is the current idiom; `ensure_future` still works but is deprecated-adjacent |

**Deprecated/outdated:**
- `websockets.serve` (top-level, legacy sync-style): Use `websockets.asyncio.server.serve` instead. The SMAIT codebase already uses the correct import in `connection/manager.py`.
- `websockets.connect` (coroutine, no reconnect): Use `async for ws in connect(uri)` for auto-reconnect.

---

## Open Questions

1. **What port does the chassis WebSocket server run on?**
   - What we know: rosbridge default is 9090; the chassis IP is 192.168.20.22
   - What's unclear: Actual port not documented in project context
   - Recommendation: Default to 9090 in `ChassisConfig`; make it configurable; confirm in Phase 13

2. **Does the chassis require a specific handshake or authentication beyond the subscribe ops?**
   - What we know: Protocol documented as subscribe/publish/call_service with no auth mentioned
   - What's unclear: Some rosbridge deployments require a handshake `{"op": "set_level", "id": "...", "level": "none"}`
   - Recommendation: Implement without auth; add auth hook in Phase 13 if needed

3. **Does `/robot_status` subscription push at a fixed rate, or only on change?**
   - What we know: Documentation says "at the configured interval" for pose
   - What's unclear: Whether status is rate-limited or event-driven
   - Recommendation: Subscribe and accept whatever rate the chassis pushes; the mock server can push at 1 Hz for tests

4. **Fragment reassembly: is `num` 0-indexed or 1-indexed?**
   - What we know: Protocol docs say `num: N, total: M`
   - What's unclear: Whether `num` starts at 0 or 1
   - Recommendation: Implement as 0-indexed (rosbridge standard); verify in Phase 13

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.x + pytest-asyncio (already installed) |
| Config file | `pyproject.toml` (no separate pytest.ini) |
| Quick run command | `python -m pytest tests/unit/test_chassis_client.py -x -q` |
| Full suite command | `python -m pytest tests/ -v --cov=smait --cov-report=term-missing` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CHAS-01 | `ChassisClient` connects to configurable IP/port | unit | `pytest tests/unit/test_chassis_client.py::TestChassisClient::test_connects_to_mock_server -x` | ❌ Wave 0 |
| CHAS-02 | Pose subscription delivers (x, y, theta) → CHASSIS_POSE_UPDATE event | unit | `pytest tests/unit/test_chassis_client.py::TestChassisClient::test_pose_subscription -x` | ❌ Wave 0 |
| CHAS-03 | Nav status → CHASSIS_NAV_STATUS event with correct status code | unit | `pytest tests/unit/test_chassis_client.py::TestChassisClient::test_nav_status_events -x` | ❌ Wave 0 |
| CHAS-04 | Robot state → CHASSIS_STATE_UPDATE event with battery/velocity | unit | `pytest tests/unit/test_chassis_client.py::TestChassisClient::test_robot_state_subscription -x` | ❌ Wave 0 |
| CHAS-05 | Soft e-stop sends advertise + publish ops to chassis | unit | `pytest tests/unit/test_chassis_client.py::TestChassisClient::test_soft_estop -x` | ❌ Wave 0 |
| CHAS-06 | Client reconnects after mock server disconnect; re-subscribes | unit | `pytest tests/unit/test_chassis_client.py::TestChassisClient::test_reconnect_on_disconnect -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/unit/test_chassis_client.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -v --cov=smait --cov-report=term-missing`
- **Phase gate:** Full suite green (131 existing + new chassis tests) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_chassis_client.py` — covers CHAS-01 through CHAS-06
- [ ] `smait/connection/chassis_client.py` — `ChassisClient` class
- [ ] `smait/core/config.py` — `ChassisConfig` dataclass + field in `Config`
- [ ] `smait/core/events.py` — 6 new `EventType` members (CHASSIS_*)

*(No new framework install needed — pytest-asyncio already in venv)*

---

## Sources

### Primary (HIGH confidence)
- websockets 16.0 installed in SMAIT venv — `connect` API, `serve` API, iterator reconnect pattern verified by running code directly in venv
- `smait/connection/manager.py` — existing WS server pattern (inbound); ChassisClient mirrors this shape outbound
- `smait/core/config.py` — existing `@dataclass` config pattern; `ChassisConfig` follows identical structure
- `smait/core/events.py` — existing `EventType(Enum)` pattern; 6 new members follow identical convention
- Protocol spec in `additional_context` — rosbridge-style JSON ops (subscribe/publish/call_service) fully documented

### Secondary (MEDIUM confidence)
- actionlib_msgs/GoalStatus status codes 0-4 (standard ROS convention; well-known but not independently verified against real chassis)
- yutong_assistance/RobotStatus nav_status codes 600-604 (from project additional_context; not independently verified)

### Tertiary (LOW confidence)
- Chassis WebSocket port: 9090 assumed from rosbridge default; actual port unknown until Phase 13 lab test

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — websockets 16.0 verified in venv; all APIs confirmed working
- Architecture: HIGH — mirrors existing `ConnectionManager` shape; in-process mock pattern confirmed working
- Pitfalls: HIGH — all pitfalls derived from verified API behavior (tested in venv) or documented codebase patterns
- Protocol details: MEDIUM — rosbridge ops verified from project documentation; nav status codes and port are assumptions

**Research date:** 2026-03-13
**Valid until:** 2026-04-13 (websockets stable; chassis protocol is hardware-defined and won't change)
