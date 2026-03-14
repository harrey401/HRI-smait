"""WayfindingManager — LLM tool registration, dispatch, and display event emission.

Owns the two OpenAI function-calling tool definitions (query_location,
navigate_to) and the async handlers that bridge between the LLM and the
navigation/map layers. Subscribes to nav outcome events for verbal
confirmations on arrival and failure.
"""

from __future__ import annotations

import logging
from typing import Callable

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.dialogue.manager import DialogueResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI tool definitions (function-calling format)
# ---------------------------------------------------------------------------

WAYFINDING_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_location",
            "description": (
                "Find where a named location is on the current floor map. "
                "Use when the user asks 'where is X?' or 'how do I get to X?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location_name": {
                        "type": "string",
                        "description": (
                            "The location name as spoken by the user "
                            "(e.g. 'ENG192', 'the bathroom', 'registration desk')"
                        ),
                    }
                },
                "required": ["location_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": (
                "Start navigating the robot to a named location. "
                "Use when the user says 'take me to X', 'go to X', or 'navigate to X'. "
                "Use after query_location confirms the location exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "poi_name": {
                        "type": "string",
                        "description": (
                            "The resolved POI name "
                            "(use after query_location confirms it exists)"
                        ),
                    }
                },
                "required": ["poi_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# WayfindingManager class
# ---------------------------------------------------------------------------


class WayfindingManager:
    """Manages wayfinding LLM tools and bridges tool calls to navigation/map layers.

    Provides get_tools() and get_tool_handlers() for registration with the
    DialogueManager. Handles DISPLAY_MAP and DISPLAY_NAV_STATUS event emission.
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        poi_kb: object,
        nav_controller: object,
        map_manager: object,
    ) -> None:
        self._cfg = config
        self._bus = event_bus
        self._poi_kb = poi_kb
        self._nav = nav_controller
        self._map = map_manager

        # Subscribe to navigation outcome events for verbal confirmation
        self._bus.subscribe(EventType.NAV_ARRIVED, self._on_nav_arrived)
        self._bus.subscribe(EventType.NAV_FAILED, self._on_nav_failed)

    def get_tools(self) -> list[dict]:
        """Return OpenAI tool definitions for registration with DialogueManager."""
        return WAYFINDING_TOOLS

    def get_tool_handlers(self) -> dict[str, Callable]:
        """Return name → async handler mapping for tool dispatch."""
        return {
            "query_location": self._handle_query_location,
            "navigate_to": self._handle_navigate_to,
        }

    async def _handle_query_location(self, args: dict) -> dict:
        """Handle query_location tool call — resolve POI and dispatch map highlight.

        Args:
            args: Tool call arguments with "location_name" key.

        Returns:
            Dict with "found" bool, optional "poi_name", and "verbal" response.
        """
        location_name = args.get("location_name", "")
        poi_name = self._poi_kb.resolve(location_name)

        if poi_name is None:
            known = self._poi_kb.list_locations()
            known_str = ", ".join(known[:5]) if known else "none"
            return {
                "found": False,
                "verbal": (
                    f"I don't know where {location_name} is. "
                    f"I know about: {known_str}."
                ),
            }

        # Render map with highlight and dispatch DISPLAY_MAP event
        png_bytes = self._map.render_map_with_highlight(poi_name)
        self._bus.emit(
            EventType.DISPLAY_MAP,
            {"png": png_bytes, "highlighted_poi": poi_name},
        )

        return {
            "found": True,
            "poi_name": poi_name,
            "verbal": f"I found {location_name} on the map!",
        }

    async def _handle_navigate_to(self, args: dict) -> dict:
        """Handle navigate_to tool call — start navigation and dispatch status.

        Args:
            args: Tool call arguments with "poi_name" key.

        Returns:
            Dict with "started" bool and "verbal" response.
        """
        poi_name = args.get("poi_name", "")
        result = await self._nav.navigate_to(poi_name)

        if result.get("success"):
            self._bus.emit(
                EventType.DISPLAY_NAV_STATUS,
                {"status": "navigating", "destination": poi_name},
            )
            return {
                "started": True,
                "verbal": f"On my way to {poi_name}!",
            }

        return {
            "started": False,
            "verbal": f"Sorry, I couldn't navigate to {poi_name}.",
        }

    def _on_nav_arrived(self, data: dict) -> None:
        """Handle NAV_ARRIVED event — emit verbal confirmation and display update.

        Emits DISPLAY_NAV_STATUS with status=arrived, then DIALOGUE_RESPONSE
        with a verbal confirmation so the robot speaks the arrival aloud.
        """
        destination = data.get("destination", "the destination")
        self._bus.emit(
            EventType.DISPLAY_NAV_STATUS,
            {"status": "arrived", "destination": destination},
        )
        self._bus.emit(
            EventType.DIALOGUE_RESPONSE,
            DialogueResponse(
                text=f"We've arrived at {destination}!",
                latency_ms=0.0,
                model_used="wayfinding",
            ),
        )

    def _on_nav_failed(self, data: dict) -> None:
        """Handle NAV_FAILED event — emit verbal explanation and display update.

        Emits DISPLAY_NAV_STATUS with status=failed, then DIALOGUE_RESPONSE
        with a verbal explanation that includes the failure reason when available.
        """
        destination = data.get("destination", "the destination")
        reason = data.get("reason", "unknown")
        self._bus.emit(
            EventType.DISPLAY_NAV_STATUS,
            {"status": "failed", "destination": destination},
        )
        verbal = f"Sorry, I wasn't able to reach {destination}."
        if reason and reason != "unknown":
            verbal += f" The issue was: {reason}."
        self._bus.emit(
            EventType.DIALOGUE_RESPONSE,
            DialogueResponse(
                text=verbal,
                latency_ms=0.0,
                model_used="wayfinding",
            ),
        )
