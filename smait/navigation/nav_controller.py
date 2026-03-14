"""NavController — orchestrates navigation using POIKnowledgeBase and ChassisClient.

Skeleton implementation — all public methods raise NotImplementedError.
Full implementation in Phase 10 Plan 04.
"""

from __future__ import annotations

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus
from smait.navigation.poi_knowledge_base import POIKnowledgeBase


class NavController:
    """Orchestrates navigation to named POIs using the chassis drive API.

    Resolves human-readable POI names via POIKnowledgeBase, calls the
    chassis /poi service, and monitors CHASSIS_NAV_STATUS events to emit
    NAV_ARRIVED / NAV_FAILED events.
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

    async def navigate_to(self, poi_name: str) -> dict:
        """Navigate to a named POI location.

        Resolves the POI name, calls the chassis /poi service, and begins
        monitoring navigation status.

        Args:
            poi_name: Chassis POI marker name (already resolved).

        Returns:
            Service response dict from the chassis /poi call.

        Raises:
            NotImplementedError: Until Plan 04 implementation.
        """
        raise NotImplementedError

    async def cancel_navigation(self) -> None:
        """Cancel any active navigation goal.

        Raises:
            NotImplementedError: Until Plan 04 implementation.
        """
        raise NotImplementedError

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
            Distance in meters.

        Raises:
            NotImplementedError: Until Plan 04 implementation.
        """
        raise NotImplementedError
