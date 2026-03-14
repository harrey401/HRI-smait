"""NavController — orchestrates navigation using POIKnowledgeBase and ChassisClient.

# SETUP-01: New Location Setup Workflow
# ======================================
# 1. Use the chassis Deployment Tool app to map the new location
#    (creates LIDAR occupancy grid stored on the chassis)
# 2. Label POI marker points using the Deployment Tool
#    (assigns chassis marker names like "eng192", "restroom_1")
# 3. Create a JSON config file at data/poi/{building_name}/{floor_name}.json
#    mapping human-friendly names to chassis marker names:
#    {"room ENG192": "eng192", "bathroom": "restroom_1"}
# 4. On startup, the system auto-detects the active floor from the chassis
#    and loads the matching JSON config (see SETUP-03: on_chassis_connected)
# 5. Users can then navigate using human-friendly names via voice commands
"""

from __future__ import annotations

import asyncio
import logging

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.navigation.poi_knowledge_base import POIKnowledgeBase

logger = logging.getLogger(__name__)


class NavController:
    """Orchestrates navigation to named POIs using the chassis drive API.

    Resolves human-readable POI names via POIKnowledgeBase, calls the
    chassis /poi service, and monitors CHASSIS_NAV_STATUS events to emit
    NAV_ARRIVED / NAV_FAILED events.

    SETUP-03: On CHASSIS_CONNECTED, auto-detects the active floor via
    /get_map_info and loads the matching POI JSON config.
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        chassis: ChassisClient,
        poi_kb: POIKnowledgeBase,
    ) -> None:
        self._cfg = config
        self._bus = event_bus
        self._chassis = chassis
        self._poi_kb = poi_kb
        self._navigating: bool = False
        self._current_destination: str = ""

        # Subscribe to nav status updates and chassis connect events
        self._bus.subscribe(EventType.CHASSIS_NAV_STATUS, self._on_nav_status)
        self._bus.subscribe(EventType.CHASSIS_CONNECTED, self.on_chassis_connected)

    @property
    def navigating(self) -> bool:
        """Return True if a navigation goal is currently active."""
        return self._navigating

    async def navigate_to(self, poi_name: str) -> dict:
        """Navigate to a named POI location.

        Resolves the POI name via POIKnowledgeBase, calls the chassis /poi
        service, and emits NAV_STARTED on success or NAV_FAILED on failure.

        Args:
            poi_name: Human-readable or chassis POI marker name.

        Returns:
            Service response dict from the chassis /poi call.
        """
        # Resolve human name to chassis marker name (fallback to original)
        marker_name = self._poi_kb.resolve(poi_name) or poi_name

        result = await self._chassis.call_service(
            "/poi", {"poi": marker_name}, timeout=10.0
        )

        if result.get("success"):
            self._navigating = True
            self._current_destination = poi_name
            self._bus.emit(EventType.NAV_STARTED, {"destination": poi_name})
        else:
            # Support protocol typo "avaliable_list" and correct spelling
            available = result.get(
                "avaliable_list", result.get("available_list", [])
            )
            self._bus.emit(
                EventType.NAV_FAILED,
                {
                    "reason": "poi_not_found",
                    "destination": poi_name,
                    "available": available,
                },
            )

        return result

    async def cancel_navigation(self) -> None:
        """Cancel any active navigation goal.

        Calls chassis send_cancel_navigation (publishes to /move_base/cancel),
        then emits NAV_CANCELLED.
        """
        await self._chassis.send_cancel_navigation()
        self._navigating = False
        self._bus.emit(EventType.NAV_CANCELLED, None)

    async def calculate_distance(
        self,
        start_x: float,
        start_y: float,
        start_floor: str,
        goal_x: float,
        goal_y: float,
        goal_floor: str,
    ) -> float:
        """Calculate the navigation distance between two poses.

        Args:
            start_x: Start x coordinate in meters.
            start_y: Start y coordinate in meters.
            start_floor: Start floor name.
            goal_x: Goal x coordinate in meters.
            goal_y: Goal y coordinate in meters.
            goal_floor: Goal floor name.

        Returns:
            Distance in meters, or -1.0 if the service call failed.
        """
        result = await self._chassis.call_service(
            "/calculate_distance",
            {
                "start_x": start_x,
                "start_y": start_y,
                "start_floor": str(start_floor),
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_floor": str(goal_floor),
            },
            timeout=15.0,
        )
        return result.get("distance", -1.0)

    def _on_nav_status(self, data: dict) -> None:
        """Handle CHASSIS_NAV_STATUS events and translate to NAV_* events.

        Status codes:
            0 = pending
            1 = active
            2 = preempted
            3 = succeeded
            4 = aborted
        """
        status = data.get("status", 0)

        if status == 3 and self._navigating:
            # Succeeded
            destination = self._current_destination
            self._navigating = False
            self._bus.emit(EventType.NAV_ARRIVED, {"destination": destination})

        elif status == 4 and self._navigating:
            # Aborted
            destination = self._current_destination
            self._navigating = False
            self._bus.emit(
                EventType.NAV_FAILED,
                {"reason": "aborted", "destination": destination},
            )

        elif status == 2 and self._navigating:
            # Preempted (cancelled externally)
            self._navigating = False
            self._bus.emit(EventType.NAV_CANCELLED, None)

    async def on_chassis_connected(self, _data) -> None:
        """SETUP-03: Auto-detect active floor and load POI config on chassis connect.

        Fetches active map info from the chassis, loads the matching POI
        JSON config, refreshes marker list, and emits MAP_ACTIVE_FLOOR.
        """
        try:
            map_info = await self._chassis.call_service(
                "/get_map_info", {"cmd": 0}, timeout=10.0
            )
            building = map_info.get("building_name", "")
            floor = map_info.get("floor_name", "")

            # Load POI config for this building/floor
            self._poi_kb.load(building, floor)

            # Fetch current markers from chassis
            await self._poi_kb.fetch_markers()

            # Emit active floor event so other components can react
            self._bus.emit(
                EventType.MAP_ACTIVE_FLOOR,
                {"building": building, "floor": floor},
            )
        except Exception:
            logger.exception("Failed to auto-detect floor on chassis connect")
