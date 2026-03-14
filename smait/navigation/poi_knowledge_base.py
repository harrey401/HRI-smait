"""POIKnowledgeBase — loads human-name → chassis-ID mappings from JSON configs.

Skeleton implementation — all public methods raise NotImplementedError.
Full implementation in Phase 10 Plan 03.
"""

from __future__ import annotations

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus


class POIKnowledgeBase:
    """Maps human-readable location names to chassis POI marker IDs.

    Loads per-floor JSON configs from the poi_config_dir, resolves human
    queries to chassis marker names, and manages marker CRUD operations.
    """

    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        chassis: ChassisClient,
    ) -> None:
        self._cfg = config.navigation
        self._bus = event_bus
        self._chassis = chassis
        self._mappings: dict[str, str] = {}
        self._chassis_markers: list[str] = []

    def load(self, building_name: str, floor_name: str) -> None:
        """Load POI name mappings from JSON config for the given building/floor.

        Args:
            building_name: Building identifier (e.g., "tefa").
            floor_name: Floor identifier (e.g., "3").

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError

    def resolve(self, human_name: str) -> str | None:
        """Resolve a human-readable location name to a chassis POI marker ID.

        Performs case-insensitive lookup in the loaded mappings.

        Args:
            human_name: Human-readable location name (e.g., "Room ENG192").

        Returns:
            Chassis marker ID string, or None if not found.

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError

    def list_locations(self) -> list[str]:
        """Return list of human-readable location names available.

        Returns:
            List of human-readable location name strings.

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError

    async def fetch_markers(self) -> list[dict]:
        """Fetch the current list of POI markers from the chassis.

        Returns:
            List of marker dicts with name, x, y, and other fields.

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError

    async def add_marker(self, name: str) -> None:
        """Insert a new POI marker at the current robot pose.

        Args:
            name: Marker name to create.

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError

    async def delete_marker(self, name: str) -> bool:
        """Delete a POI marker by name.

        Args:
            name: Marker name to delete.

        Returns:
            True if deletion succeeded, False otherwise.

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError

    def update_chassis_markers(self, marker_names: list[str]) -> None:
        """Update the locally cached list of chassis marker names.

        Args:
            marker_names: List of marker name strings from the chassis.

        Raises:
            NotImplementedError: Until Plan 03 implementation.
        """
        raise NotImplementedError
