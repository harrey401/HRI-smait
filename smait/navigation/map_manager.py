"""MapManager — subscribes to map topic, decodes PNG tiles, renders overlays.

Skeleton implementation — all public methods raise NotImplementedError.
Full implementation in Phase 10 Plan 02.
"""

from __future__ import annotations

from PIL import Image

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus


def world_to_pixel(wx: float, wy: float, meta: dict) -> tuple[int, int]:
    """Convert world coordinates (meters) to pixel coordinates.

    Args:
        wx: World x coordinate in meters.
        wy: World y coordinate in meters.
        meta: Map metadata dict with keys: origin_x, origin_y, resolution, height.

    Returns:
        (px, py) pixel coordinates.
    """
    raise NotImplementedError


def draw_robot_arrow(
    draw,
    px: int,
    py: int,
    theta: float,
    length: int,
    color: str,
) -> None:
    """Draw a directional arrow representing the robot's pose on the map.

    Args:
        draw: PIL ImageDraw instance.
        px: Robot pixel x coordinate.
        py: Robot pixel y coordinate.
        theta: Robot heading in radians.
        length: Arrow length in pixels.
        color: Arrow color string.
    """
    raise NotImplementedError


class MapManager:
    """Manages map tiles received from the chassis, renders overlays for display.

    Subscribes to the map topic via ChassisClient, decodes base64 PNG fragments,
    assembles tiles, and overlays robot pose and path data.
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
        self._map_image: Image.Image | None = None
        self._map_meta: dict = {}
        self._current_pose: dict = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self._path_points: list[tuple[float, float]] = []

    async def start(self) -> None:
        """Subscribe to map topic and wire event handlers.

        Raises:
            NotImplementedError: Until Plan 02 implementation.
        """
        raise NotImplementedError

    async def list_maps(self) -> list[dict]:
        """Retrieve the list of available maps from the chassis.

        Returns:
            List of map info dicts with building_name and floor_info keys.

        Raises:
            NotImplementedError: Until Plan 02 implementation.
        """
        raise NotImplementedError

    async def switch_map(self, building: str, floor: str) -> bool:
        """Switch the active map to the specified building/floor.

        Args:
            building: Building name (e.g., "tefa").
            floor: Floor name/number (e.g., "3").

        Returns:
            True if switch succeeded, False otherwise.

        Raises:
            NotImplementedError: Until Plan 02 implementation.
        """
        raise NotImplementedError

    async def get_active_map_info(self) -> dict:
        """Get metadata about the currently active map.

        Returns:
            Dict with building, floor, origin, resolution keys.

        Raises:
            NotImplementedError: Until Plan 02 implementation.
        """
        raise NotImplementedError

    def render_map(self) -> bytes:
        """Render the current map with robot pose and path overlay as PNG bytes.

        Returns:
            PNG image bytes.

        Raises:
            NotImplementedError: Until Plan 02 implementation.
        """
        raise NotImplementedError
