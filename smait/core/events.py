"""Async EventBus pub/sub — ALL inter-module communication."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    # Audio
    SPEECH_DETECTED = auto()
    SPEECH_SEGMENT = auto()

    # Separation
    SPEECH_SEPARATED = auto()

    # ASR
    TRANSCRIPT_READY = auto()
    TRANSCRIPT_REJECTED = auto()

    # Turn-taking
    END_OF_TURN = auto()
    BARGE_IN = auto()

    # Vision
    FACE_DETECTED = auto()
    FACE_LOST = auto()
    FACE_UPDATED = auto()
    GAZE_UPDATE = auto()
    LIP_ROI_READY = auto()

    # Engagement
    ENGAGEMENT_START = auto()
    ENGAGEMENT_LOST = auto()

    # Dialogue
    DIALOGUE_RESPONSE = auto()
    DIALOGUE_STREAM = auto()

    # TTS
    TTS_START = auto()
    TTS_END = auto()
    TTS_AUDIO_CHUNK = auto()

    # Session
    SESSION_START = auto()
    SESSION_END = auto()

    # Hardware / Connection
    DOA_UPDATE = auto()
    CAE_STATUS = auto()
    CONNECTION_OPEN = auto()
    CONNECTION_CLOSED = auto()

    # VAD
    VAD_PROB = auto()

    # System
    ERROR = auto()

    # Chassis / Navigation
    CHASSIS_POSE_UPDATE = auto()      # data: {"x": float, "y": float, "theta": float}
    CHASSIS_NAV_STATUS = auto()       # data: {"status": int, "text": str, "goal_id": str}
    CHASSIS_STATE_UPDATE = auto()     # data: {"battery": float, "nav_status": int, "control_state": int, "velocity": list}
    CHASSIS_OBSTACLE = auto()         # data: {"region": int}
    CHASSIS_CONNECTED = auto()        # data: None
    CHASSIS_DISCONNECTED = auto()     # data: None


class EventBus:
    """Async pub/sub. Supports both coroutine and sync handlers."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable]] = {}

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler.__qualname__, event_type.name)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Remove a handler for an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    def emit(self, event_type: EventType, data: Any = None) -> None:
        """Emit an event, scheduling async handlers on the running loop.

        Safe to call from both sync and async contexts.
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — call sync handlers directly
            for handler in handlers:
                if not asyncio.iscoroutinefunction(handler):
                    try:
                        handler(data)
                    except Exception:
                        logger.exception("Error in sync handler %s for %s",
                                         handler.__qualname__, event_type.name)
            return

        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                loop.create_task(self._safe_async_call(handler, event_type, data))
            else:
                try:
                    handler(data)
                except Exception:
                    logger.exception("Error in sync handler %s for %s",
                                     handler.__qualname__, event_type.name)

    async def emit_async(self, event_type: EventType, data: Any = None) -> None:
        """Emit an event and await all async handlers."""
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(self._safe_async_call(handler, event_type, data))
            else:
                try:
                    handler(data)
                except Exception:
                    logger.exception("Error in sync handler %s for %s",
                                     handler.__qualname__, event_type.name)

        if tasks:
            await asyncio.gather(*tasks)

    @staticmethod
    async def _safe_async_call(
        handler: Callable, event_type: EventType, data: Any
    ) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Error in async handler %s for %s",
                             handler.__qualname__, event_type.name)
