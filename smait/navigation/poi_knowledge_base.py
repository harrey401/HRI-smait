"""POIKnowledgeBase — loads human-name → chassis-ID mappings from JSON configs.

Implements marker CRUD operations via ChassisClient and case-insensitive
name resolution from per-floor JSON config files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)


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

        Reads data/poi/{building_name}/{floor_name}.json and populates
        self._mappings with lowercase-keyed human name → chassis marker name
        entries. Emits POI_CONFIG_MISSING if the file does not exist.

        Args:
            building_name: Building identifier (e.g., "tefa").
            floor_name: Floor identifier (e.g., "3").
        """
        config_path = Path(self._cfg.poi_config_dir) / building_name / f"{floor_name}.json"

        if not config_path.exists():
            logger.warning(
                "POI config not found: %s (building=%s, floor=%s)",
                config_path, building_name, floor_name,
            )
            self._mappings = {}
            self._bus.emit(
                EventType.POI_CONFIG_MISSING,
                {"building": building_name, "floor": floor_name},
            )
            return

        with open(config_path) as f:
            raw: dict[str, str] = json.load(f)

        # Normalize keys to lowercase for case-insensitive resolution
        self._mappings = {k.lower(): v for k, v in raw.items()}
        logger.debug(
            "Loaded %d POI mappings from %s", len(self._mappings), config_path
        )

    def resolve(self, human_name: str) -> str | None:
        """Resolve a human-readable location name to a chassis POI marker ID.

        Performs case-insensitive lookup against loaded mappings first, then
        falls back to checking whether the name exists verbatim in the known
        chassis marker list.

        Args:
            human_name: Human-readable location name (e.g., "Room ENG192").

        Returns:
            Chassis marker ID string, or None if not found.
        """
        lower = human_name.lower()
        for key, value in self._mappings.items():
            if key.lower() == lower:
                return value

        if human_name in self._chassis_markers:
            return human_name

        return None

    def list_locations(self) -> list[str]:
        """Return sorted list of all known location names.

        Includes both human-readable names from loaded config and raw chassis
        marker names fetched from the robot.

        Returns:
            Sorted list of location name strings.
        """
        combined = set(self._mappings.keys()) | set(self._chassis_markers)
        return sorted(combined)

    def update_chassis_markers(self, marker_names: list[str]) -> None:
        """Update the locally cached list of chassis marker names.

        Args:
            marker_names: List of marker name strings from the chassis.
        """
        self._chassis_markers = list(marker_names)

    async def fetch_markers(self) -> list[dict]:
        """Fetch the current list of POI markers from the chassis.

        Calls /marker_operation/get_markers, extracts the waypoints list,
        updates the local chassis marker cache, and emits POI_LIST_UPDATED.

        Returns:
            List of marker dicts (each has at least a "name" key).
        """
        result = await self._chassis.call_service("/marker_operation/get_markers")

        # Guard: result must be a dict (handles mock/error cases gracefully)
        if not isinstance(result, dict):
            logger.warning("fetch_markers: unexpected response type %s", type(result))
            return []

        # Support both flat {"waypoints": [...]} and nested {"markers": {"waypoints": [...]}}
        if "waypoints" in result:
            waypoints: list[dict] = result["waypoints"]
        else:
            waypoints = result.get("markers", {}).get("waypoints", [])

        marker_names = [wp["name"] for wp in waypoints if "name" in wp]
        self.update_chassis_markers(marker_names)

        self._bus.emit(EventType.POI_LIST_UPDATED, {"markers": waypoints})
        return waypoints

    async def add_marker(self, name: str) -> None:
        """Insert a new POI marker at the current robot pose.

        Calls send_insert_marker on the chassis, then refreshes the local
        marker list via fetch_markers.

        Args:
            name: Marker name to create.
        """
        await self._chassis.send_insert_marker(name)
        await self.fetch_markers()

    async def delete_marker(self, name: str) -> bool:
        """Delete a POI marker by name.

        Calls /marker_manager/delete_poi on the chassis. Handles the protocol
        typo "avaliable_list" (checking both spellings defensively). Updates
        the local marker cache and emits POI_LIST_UPDATED.

        Args:
            name: Marker name to delete.

        Returns:
            True if the chassis reported success, False otherwise.
        """
        result = await self._chassis.call_service(
            "/marker_manager/delete_poi", {"poi": name}
        )

        success: bool = bool(result.get("success", False))

        # Protocol uses "avaliable_list" (typo) — check both spellings defensively
        remaining: list[str] = result.get(
            "avaliable_list", result.get("available_list", [])
        )
        if remaining:
            self.update_chassis_markers(remaining)

        self._bus.emit(EventType.POI_LIST_UPDATED, {"markers": remaining})
        return success
