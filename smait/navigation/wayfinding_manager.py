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
        raise NotImplementedError

    async def _handle_navigate_to(self, args: dict) -> dict:
        """Handle navigate_to tool call — start navigation and dispatch status.

        Args:
            args: Tool call arguments with "poi_name" key.

        Returns:
            Dict with "started" bool and "verbal" response.
        """
        raise NotImplementedError
