"""MapManager — subscribes to map topic, decodes PNG tiles, renders overlays.

Manages map images received from the chassis via the op:png protocol.
Decodes base64 PNG data, renders robot pose and path overlays, and
handles map list/switch operations via chassis service calls.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math

from PIL import Image, ImageDraw

from smait.connection.chassis_client import ChassisClient
from smait.core.config import Config
from smait.core.events import EventBus, EventType

logger = logging.getLogger(__name__)

# Default placeholder map size when no map has been received
_PLACEHOLDER_SIZE = (100, 100)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def world_to_pixel(wx: float, wy: float, meta: dict) -> tuple[int, int]:
    """Convert world coordinates (meters) to pixel coordinates.

    Uses the standard ROS OccupancyGrid convention: origin is the bottom-left
    corner of the map, Y increases upward in world space but downward in image
    space (hence the row flip).

    Args:
        wx: World x coordinate in meters.
        wy: World y coordinate in meters.
        meta: Map metadata dict with keys: origin_x, origin_y, resolution,
              width, height.

    Returns:
        (col, row) pixel coordinates clamped to image bounds.
    """
    origin_x: float = meta["origin_x"]
    origin_y: float = meta["origin_y"]
    resolution: float = meta["resolution"]
    width: int = meta["width"]
    height: int = meta["height"]

    col = int((wx - origin_x) / resolution)
    row = int(height - (wy - origin_y) / resolution)

    # Clamp to valid image bounds
    col = max(0, min(col, width - 1))
    row = max(0, min(row, height - 1))

    return (col, row)


def draw_robot_arrow(
    draw: ImageDraw.ImageDraw,
    px: int,
    py: int,
    theta: float,
    length: int = 15,
    color: str = "red",
) -> None:
    """Draw a directional arrow representing the robot's pose on the map.

    Draws a filled circle at the robot position and a line indicating heading.

    Args:
        draw: PIL ImageDraw instance.
        px: Robot pixel x (column) coordinate.
        py: Robot pixel y (row) coordinate.
        theta: Robot heading in radians (0 = positive X, CCW positive).
        length: Arrow length in pixels.
        color: Arrow and body color string.
    """
    tip_x = px + int(length * math.cos(theta))
    tip_y = py - int(length * math.sin(theta))  # negate: image Y is downward

    draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color)
    draw.line([px, py, tip_x, tip_y], fill=color, width=3)


def decode_map_png(msg: dict) -> tuple[Image.Image, dict]:
    """Decode a map PNG message into a PIL Image and metadata dict.

    Supports two message formats:
    - Flat: {"data": "<b64>", "width": int, "height": int, "origin_x": float, ...}
    - Nested (rosbridge op:png): {"msg": {"data": "<b64>", "info": {"origin": ..., ...}}}

    Args:
        msg: Map message dict.

    Returns:
        (image, meta) where meta has keys: origin_x, origin_y, resolution,
        width, height.
    """
    nested = msg.get("msg")
    if nested is not None:
        # Rosbridge nested format
        info = nested.get("info", {})
        origin = info.get("origin", {}).get("position", {})
        origin_x = float(origin.get("x", 0.0))
        origin_y = float(origin.get("y", 0.0))
        resolution = float(info.get("resolution", 0.05))
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        png_b64 = nested.get("data", "")
    else:
        # Flat format used in tests
        origin_x = float(msg.get("origin_x", 0.0))
        origin_y = float(msg.get("origin_y", 0.0))
        resolution = float(msg.get("resolution", 0.05))
        width = int(msg.get("width", 0))
        height = int(msg.get("height", 0))
        png_b64 = msg.get("data", "")

    png_bytes = base64.b64decode(png_b64)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    meta = {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "resolution": resolution,
        "width": width or image.width,
        "height": height or image.height,
    }
    return image, meta


# ---------------------------------------------------------------------------
# MapManager class
# ---------------------------------------------------------------------------


class MapManager:
    """Manages map tiles received from the chassis, renders overlays for display.

    Subscribes to the map topic via ChassisClient, decodes base64 PNG fragments,
    and overlays robot pose and path data. Emits MAP_RENDERED on each update.
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
        self._poi_positions: dict[str, dict] = {}  # name → {"x": float, "y": float}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to map topic and wire event handlers.

        Registers listeners on the EventBus for chassis map, pose, path, and
        connection events. Sends the map subscription request to the chassis.
        """
        self._bus.subscribe(EventType.CHASSIS_MAP_UPDATE, self._on_map_update)
        self._bus.subscribe(EventType.CHASSIS_POSE_UPDATE, self._on_pose_update)
        self._bus.subscribe(EventType.CHASSIS_PATH_UPDATE, self._on_path_update)
        self._bus.subscribe(EventType.CHASSIS_CONNECTED, self._on_chassis_connected)
        self._bus.subscribe(EventType.POI_LIST_UPDATED, self._on_poi_list_updated)

        await self._chassis.subscribe_topic(
            self._cfg.map_topic,
            "nav_msgs/OccupancyGrid",
            compression="png",
            fragment_size=self._cfg.map_fragment_size,
        )

    # ------------------------------------------------------------------
    # Map service operations
    # ------------------------------------------------------------------

    async def list_maps(self) -> list[dict]:
        """Retrieve the list of available maps from the chassis.

        Calls /layered_map_cmd op=0 and returns the list_info array.

        Returns:
            List of map info dicts with building_name and floor_info keys.
        """
        result = await self._chassis.call_service("/layered_map_cmd", {"op": 0})
        return result.get("list_info", [])

    async def switch_map(self, building: str, floor: str) -> bool:
        """Switch the active map to the specified building/floor.

        Args:
            building: Building name (e.g., "tefa").
            floor: Floor name/number (e.g., "3").

        Returns:
            True if switch succeeded (result == 0), False otherwise.
        """
        result = await self._chassis.call_service(
            "/node_manager_control",
            {
                "building_name": building,
                "floor_num": str(floor),
                "cmd": 7,
                "args": 0,
            },
        )
        return result.get("result") == 0

    async def get_active_map_info(self) -> dict:
        """Get metadata about the currently active map.

        Returns:
            Dict from /get_map_info service response.
        """
        result = await self._chassis.call_service("/get_map_info", {"cmd": 0})
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_map(self) -> bytes:
        """Render the current map with robot pose and path overlay as PNG bytes.

        When no map has been received yet, returns a placeholder grey PNG.

        Returns:
            PNG image bytes.
        """
        if self._map_image is not None:
            base = self._map_image.copy()
            meta = self._map_meta
        else:
            # Placeholder: grey image, unit meta so world_to_pixel works
            base = Image.new("RGBA", _PLACEHOLDER_SIZE, color=(200, 200, 200, 255))
            meta = {
                "origin_x": 0.0,
                "origin_y": 0.0,
                "resolution": 1.0,
                "width": _PLACEHOLDER_SIZE[0],
                "height": _PLACEHOLDER_SIZE[1],
            }

        draw = ImageDraw.Draw(base)

        # Draw path polyline if we have 2+ points
        if len(self._path_points) >= 2:
            pixel_path = [
                world_to_pixel(x, y, meta) for x, y in self._path_points
            ]
            draw.line(pixel_path, fill=self._cfg.path_color, width=2)

        # Draw robot pose arrow
        px, py = world_to_pixel(
            self._current_pose["x"],
            self._current_pose["y"],
            meta,
        )
        draw_robot_arrow(
            draw,
            px,
            py,
            self._current_pose.get("theta", 0.0),
            length=self._cfg.arrow_length_px,
            color=self._cfg.arrow_color,
        )

        buf = io.BytesIO()
        base.save(buf, format="PNG")
        return buf.getvalue()

    def render_map_with_highlight(
        self,
        poi_name: str,
        highlight_color: str | None = None,
        radius: int | None = None,
    ) -> bytes:
        """Render the map with a destination POI highlighted as a circle overlay.

        Draws the highlight circle before the robot arrow so the arrow renders
        on top. Falls back to the base render when no map is loaded.

        Args:
            poi_name: Chassis marker name whose position should be highlighted.
            highlight_color: PIL color string for the circle. Defaults to config value.
            radius: Highlight circle radius in pixels. Defaults to config value.

        Returns:
            PNG image bytes with path, highlight circle, and robot arrow overlaid.
        """
        if highlight_color is None:
            highlight_color = self._cfg.highlight_color
        if radius is None:
            radius = self._cfg.highlight_radius_px

        if self._map_image is not None:
            base = self._map_image.copy()
            meta = self._map_meta
        else:
            base = Image.new("RGBA", _PLACEHOLDER_SIZE, color=(200, 200, 200, 255))
            meta = {
                "origin_x": 0.0,
                "origin_y": 0.0,
                "resolution": 1.0,
                "width": _PLACEHOLDER_SIZE[0],
                "height": _PLACEHOLDER_SIZE[1],
            }

        draw = ImageDraw.Draw(base)

        # Draw destination highlight circle
        poi_pos = self._poi_positions.get(poi_name)
        if poi_pos:
            hx, hy = world_to_pixel(poi_pos["x"], poi_pos["y"], meta)
            draw.ellipse(
                [hx - radius, hy - radius, hx + radius, hy + radius],
                fill=highlight_color,
                outline="orange",
                width=2,
            )

        # Draw path polyline if we have 2+ points
        if len(self._path_points) >= 2:
            pixel_path = [
                world_to_pixel(x, y, meta) for x, y in self._path_points
            ]
            draw.line(pixel_path, fill=self._cfg.path_color, width=2)

        # Draw robot pose arrow
        px, py = world_to_pixel(
            self._current_pose["x"],
            self._current_pose["y"],
            meta,
        )
        draw_robot_arrow(
            draw,
            px,
            py,
            self._current_pose.get("theta", 0.0),
            length=self._cfg.arrow_length_px,
            color=self._cfg.arrow_color,
        )

        buf = io.BytesIO()
        base.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Private event handlers
    # ------------------------------------------------------------------

    def _on_poi_list_updated(self, data: dict) -> None:
        """Handle POI_LIST_UPDATED — cache POI world positions for highlight rendering.

        Populates _poi_positions from marker data. Supports both flat {x, y}
        and nested {pose: {position: {x, y}}} coordinate formats.
        """
        markers = data.get("markers", [])
        new_positions: dict[str, dict] = {}
        for marker in markers:
            name = marker.get("name")
            if not name:
                continue
            # Support flat coordinates
            if "x" in marker and "y" in marker:
                new_positions[name] = {"x": float(marker["x"]), "y": float(marker["y"])}
            else:
                # Support nested pose.position.x/y
                pose = marker.get("pose", {})
                position = pose.get("position", {})
                if "x" in position and "y" in position:
                    new_positions[name] = {
                        "x": float(position["x"]),
                        "y": float(position["y"]),
                    }
        self._poi_positions = new_positions

    def _on_map_update(self, data: dict) -> None:
        """Handle CHASSIS_MAP_UPDATE — decode PNG and re-render."""
        try:
            image, meta = decode_map_png(data)
            self._map_image = image
            self._map_meta = meta
            png_bytes = self.render_map()
            self._bus.emit(EventType.MAP_RENDERED, png_bytes)
        except Exception:
            logger.exception("MapManager: failed to decode map update")

    def _on_pose_update(self, data: dict) -> None:
        """Handle CHASSIS_POSE_UPDATE — update pose and re-render if map available."""
        self._current_pose = {
            "x": data.get("x", 0.0),
            "y": data.get("y", 0.0),
            "theta": data.get("theta", 0.0),
        }
        if self._map_image is not None:
            try:
                png_bytes = self.render_map()
                self._bus.emit(EventType.MAP_RENDERED, png_bytes)
            except Exception:
                logger.exception("MapManager: failed to render on pose update")

    def _on_path_update(self, data: dict) -> None:
        """Handle CHASSIS_PATH_UPDATE — update path and re-render if map available."""
        self._path_points = data.get("points", [])
        if self._map_image is not None:
            try:
                png_bytes = self.render_map()
                self._bus.emit(EventType.MAP_RENDERED, png_bytes)
            except Exception:
                logger.exception("MapManager: failed to render on path update")

    async def _on_chassis_connected(self, _data: object) -> None:
        """Handle CHASSIS_CONNECTED — auto-detect active map and available maps.

        Queries /get_map_info and /layered_map_cmd concurrently, then emits
        MAP_ACTIVE_FLOOR and MAP_LIST_UPDATED events.
        """
        try:
            map_info_result, list_result = await asyncio.gather(
                self._chassis.call_service("/get_map_info", {"cmd": 0}),
                self._chassis.call_service("/layered_map_cmd", {"op": 0}),
                return_exceptions=True,
            )

            # Emit MAP_ACTIVE_FLOOR from map_info response
            if isinstance(map_info_result, dict):
                building = None
                floor = None

                # Try flat keys first
                if "building_name" in map_info_result:
                    building = map_info_result.get("building_name")
                    floor = map_info_result.get("floor_name")
                elif "list_info" in map_info_result:
                    # list_info format: [{building_name, floor_info: [{floor_name, current}]}]
                    list_info = map_info_result["list_info"]
                    if list_info:
                        entry = list_info[0]
                        building = entry.get("building_name")
                        floors = entry.get("floor_info", [])
                        # Find current floor or fall back to first
                        current = next((f for f in floors if f.get("current")), None)
                        if current:
                            floor = current.get("floor_name")
                        elif floors:
                            floor = floors[0].get("floor_name")

                if building:
                    self._bus.emit(
                        EventType.MAP_ACTIVE_FLOOR,
                        {"building": building, "floor": floor or ""},
                    )

            # Emit MAP_LIST_UPDATED from list_maps response
            if isinstance(list_result, dict):
                self._bus.emit(
                    EventType.MAP_LIST_UPDATED,
                    {"list_info": list_result.get("list_info", [])},
                )

        except Exception:
            logger.exception("MapManager: auto-detect on chassis connected failed")
