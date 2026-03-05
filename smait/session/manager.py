"""Session lifecycle state machine and proactive behaviors."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum, auto
from typing import Any, Optional

from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

PROACTIVE_GREETINGS = [
    "Hi there! I'm Jackie. What brings you to the conference today?",
    "Hey! Welcome! I'm Jackie, your friendly conference robot. What's on your mind?",
    "Hello! I spotted you looking my way. I'm Jackie — anything I can help with?",
]


class SessionState(Enum):
    IDLE = auto()
    APPROACHING = auto()
    ENGAGED = auto()
    CONVERSING = auto()
    DISENGAGING = auto()


class SessionManager:
    """Orchestrates the full interaction lifecycle.

    State machine: IDLE -> APPROACHING -> ENGAGED -> CONVERSING -> DISENGAGING -> IDLE

    Subscribes to engagement, turn-taking, dialogue, TTS, face, and connection events.
    On each state change, sends state JSON to Jackie for UI update.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.session
        self._event_bus = event_bus
        self._state = SessionState.IDLE
        self._session_id: Optional[str] = None
        self._target_track_id: Optional[int] = None
        self._session_start_time: Optional[float] = None
        self._last_activity_time: Optional[float] = None
        self._greeting_index = 0

        # Grace timers
        self._face_lost_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None

        # Subscribe to events
        event_bus.subscribe(EventType.ENGAGEMENT_START, self._on_engagement_start)
        event_bus.subscribe(EventType.ENGAGEMENT_LOST, self._on_engagement_lost)
        event_bus.subscribe(EventType.FACE_LOST, self._on_face_lost)
        event_bus.subscribe(EventType.FACE_DETECTED, self._on_face_detected)
        event_bus.subscribe(EventType.TRANSCRIPT_READY, self._on_transcript_ready)
        event_bus.subscribe(EventType.DIALOGUE_RESPONSE, self._on_dialogue_response)
        event_bus.subscribe(EventType.TTS_END, self._on_tts_end)
        event_bus.subscribe(EventType.SESSION_END, self._on_session_end)
        event_bus.subscribe(EventType.CONNECTION_CLOSED, self._on_connection_closed)

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def target_track_id(self) -> Optional[int]:
        return self._target_track_id

    async def check_timeouts(self) -> None:
        """Periodic check for session timeouts. Call from main loop."""
        now = time.monotonic()

        if self._state == SessionState.IDLE:
            return

        # Face lost grace period
        if self._face_lost_time is not None:
            elapsed = now - self._face_lost_time
            if elapsed >= self._config.face_lost_grace_s:
                logger.info("Face lost grace period expired (%.1fs)", elapsed)
                await self._transition(SessionState.IDLE, reason="face_lost_timeout")
                return

        # Silence timeout
        if self._state == SessionState.CONVERSING and self._last_activity_time is not None:
            silence = now - self._last_activity_time
            if silence >= self._config.timeout_s:
                logger.info("Silence timeout (%.1fs)", silence)
                await self._transition(SessionState.DISENGAGING, reason="silence_timeout")
                # Auto-transition to IDLE after farewell
                await self._farewell()
                await self._transition(SessionState.IDLE, reason="silence_timeout")
                return

    async def _on_engagement_start(self, data: Any) -> None:
        """Handle engagement detection."""
        if self._state not in (SessionState.IDLE, SessionState.APPROACHING):
            return

        track_id = data.get("track_id") if isinstance(data, dict) else None
        self._target_track_id = track_id
        self._face_lost_time = None

        await self._transition(SessionState.ENGAGED)

        # Proactive greeting
        greeting = PROACTIVE_GREETINGS[self._greeting_index % len(PROACTIVE_GREETINGS)]
        self._greeting_index += 1

        self._event_bus.emit(EventType.SESSION_START, {
            "session_id": self._session_id,
            "track_id": track_id,
            "greeting": greeting,
        })

        # Immediately transition to conversing
        await self._transition(SessionState.CONVERSING)

    async def _on_engagement_lost(self, data: Any) -> None:
        """Handle engagement loss."""
        if self._state in (SessionState.IDLE, SessionState.APPROACHING):
            return

        await self._transition(SessionState.DISENGAGING, reason="engagement_lost")
        await self._farewell()
        await self._transition(SessionState.IDLE, reason="engagement_lost")

    async def _on_face_lost(self, data: Any) -> None:
        """Handle face disappearing — start grace timer."""
        if self._state == SessionState.IDLE:
            return

        if isinstance(data, dict):
            lost_track_id = data.get("track_id")
            if lost_track_id == self._target_track_id:
                self._face_lost_time = time.monotonic()
                logger.info("Target face lost, starting %.1fs grace period",
                            self._config.face_lost_grace_s)

    async def _on_face_detected(self, data: Any) -> None:
        """Handle face reappearing — check for reacquisition."""
        if self._face_lost_time is None:
            return

        if isinstance(data, dict):
            track = data.get("track")
            if track is not None:
                # Check if this could be the same person returning
                elapsed = time.monotonic() - self._face_lost_time
                if elapsed <= self._config.reacquisition_window_s:
                    # Re-associate: assume same person returned
                    if hasattr(track, "track_id"):
                        self._target_track_id = track.track_id
                        self._face_lost_time = None
                        logger.info("Face reacquired (track_id=%d, after %.1fs)",
                                    track.track_id, elapsed)

    async def _on_transcript_ready(self, data: Any) -> None:
        """Handle completed transcript — route to dialogue."""
        self._last_activity_time = time.monotonic()

    async def _on_dialogue_response(self, data: Any) -> None:
        """Handle LLM response."""
        self._last_activity_time = time.monotonic()

    async def _on_tts_end(self, _data: Any) -> None:
        """Handle TTS completion."""
        self._last_activity_time = time.monotonic()

    async def _on_session_end(self, data: Any) -> None:
        """Handle explicit session end (e.g., goodbye detection)."""
        if self._state == SessionState.IDLE:
            return

        reason = data.get("reason", "explicit") if isinstance(data, dict) else "explicit"
        logger.info("Session end requested: %s", reason)

        if self._state != SessionState.DISENGAGING:
            await self._transition(SessionState.DISENGAGING, reason=reason)

        await self._transition(SessionState.IDLE, reason=reason)

    async def _on_connection_closed(self, _data: Any) -> None:
        """Handle Jackie disconnection."""
        if self._state != SessionState.IDLE:
            await self._transition(SessionState.IDLE, reason="connection_closed")

    async def _transition(self, new_state: SessionState, reason: str = "") -> None:
        """Transition to a new session state."""
        old_state = self._state
        self._state = new_state

        logger.info("Session: %s -> %s%s",
                     old_state.name, new_state.name,
                     f" ({reason})" if reason else "")

        # Session lifecycle
        if new_state == SessionState.ENGAGED and old_state == SessionState.IDLE:
            self._session_id = str(uuid.uuid4())[:8]
            self._session_start_time = time.monotonic()
            self._last_activity_time = time.monotonic()

        if new_state == SessionState.IDLE and old_state != SessionState.IDLE:
            self._cleanup_session()

        # Notify Jackie of state change
        jackie_state = "engaged" if new_state != SessionState.IDLE else "idle"
        robot_status = {
            SessionState.IDLE: "listening",
            SessionState.APPROACHING: "listening",
            SessionState.ENGAGED: "listening",
            SessionState.CONVERSING: "listening",
            SessionState.DISENGAGING: "speaking",
        }.get(new_state, "listening")

        # Emit event for ConnectionManager to forward to Jackie
        self._event_bus.emit(EventType.CONNECTION_OPEN, {
            "action": "send_state",
            "state": jackie_state,
            "robot_status": robot_status,
        })

    async def _farewell(self) -> None:
        """Generate a farewell message."""
        self._event_bus.emit(EventType.DIALOGUE_RESPONSE, {
            "text": "It was great chatting with you! Enjoy the conference!",
            "is_farewell": True,
        })

    def _cleanup_session(self) -> None:
        """Clean up session state."""
        if self._session_id and self._session_start_time:
            duration = time.monotonic() - self._session_start_time
            logger.info("Session %s ended (duration=%.1fs)", self._session_id, duration)

        self._session_id = None
        self._target_track_id = None
        self._session_start_time = None
        self._last_activity_time = None
        self._face_lost_time = None
        self._silence_start_time = None
