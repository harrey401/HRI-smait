#!/usr/bin/env python3
"""SMAIT HRI v3.0 — Entry point for the Jackie robot interaction system.

Usage:
    python run_jackie.py [OPTIONS]

Options:
    --host HOST          WebSocket bind address (default: 0.0.0.0)
    --port PORT          WebSocket port (default: 8765)
    --voice-only         Audio pipeline only, no camera/vision
    --debug              Enable debug mode
    --show-video         Show video feed with face tracking overlay
    --config FILE        Path to JSON config file
    --event-name NAME    Event name for log directory (default: hfes)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# Disable CUDA graphs for Blackwell (sm_120) / RTX 5070
os.environ.setdefault("NEMO_DISABLE_CUDA_GRAPHS", "1")

# Use uvloop for better async performance on Linux
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("smait.log", mode="a"),
        ],
    )
    # Quiet noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("nemo").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMAIT HRI v3.0 — Jackie Robot Interaction System",
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--voice-only", action="store_true",
                        help="Audio pipeline only, no camera/vision")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--show-video", action="store_true",
                        help="Show video feed with face tracking overlay")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--event-name", default="hfes",
                        help="Event name for log directory")
    return parser.parse_args()


def get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


async def run(args: argparse.Namespace) -> None:
    from smait.core.config import Config, get_config, reset_config
    from smait.main import HRISystem

    # Load config
    reset_config()
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config()

    # Apply CLI overrides
    config.connection.host = args.host
    config.connection.port = args.port
    config.debug = args.debug
    if args.show_video:
        config.show_video = True  # CLI flag enables; config default (True) used otherwise

    local_ip = get_local_ip()
    logger = logging.getLogger("smait")
    logger.info("=" * 60)
    logger.info("  SMAIT HRI v3.0 — Jackie Robot Interaction System")
    logger.info("=" * 60)
    logger.info("  Connect Jackie to: ws://%s:%d", local_ip, config.connection.port)
    logger.info("  Mode: %s", "voice-only" if args.voice_only else "full (audio + vision)")
    logger.info("  Debug: %s", config.debug)
    logger.info("  Event: %s", args.event_name)
    logger.info("=" * 60)

    system = HRISystem(config, voice_only=args.voice_only)
    await system.run()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logging.getLogger("smait").info("Shutting down (Ctrl+C)")


if __name__ == "__main__":
    main()
