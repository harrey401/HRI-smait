#!/usr/bin/env python3
"""SMAIT HRI v2.0 - Jackie Robot Mode"""
import argparse
import sys
import os
import signal
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress Qt/wayland warnings (cosmetic, not errors)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--voice-only", action="store_true",
                        help="Skip face detection — listen immediately (use when camera unavailable)")
    args = parser.parse_args()
    
    local_ip = get_local_ip()
    print(f"\n{'='*60}")
    print(f"  SMAIT HRI v2.0 - Jackie Mode")
    print(f"  Connect Jackie to: ws://{local_ip}:{args.port}")
    print(f"{'='*60}\n")
    
    # Initialize Jackie sources FIRST
    from smait.sensors.network_source import init_jackie_sources, shutdown_jackie
    audio_source, video_source = init_jackie_sources(host=args.host, port=args.port)
    
    # Register sources with factory functions BEFORE importing HRISystem
    from smait.sensors.sources import set_jackie_audio_source, set_jackie_video_source
    set_jackie_audio_source(audio_source)
    set_jackie_video_source(video_source)
    
    def cleanup(sig=None, frame=None):
        print("\n[SMAIT] Shutting down...")
        shutdown_jackie()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)
    
    if args.test:
        print("[TEST MODE] Press Ctrl+C to exit\n")
        import cv2, time
        try:
            while True:
                ret, frame = video_source.read()
                if ret and frame is not None:
                    cv2.imshow("Jackie Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()
        cleanup()
        return
    
    # Now import HRISystem - it will use our Jackie sources
    from smait.main import HRISystem, set_jackie_server
    from smait.core.config import get_config
    from smait.sensors.network_source import get_jackie_server
    
    # Wire Jackie server into main for sending messages back to app
    set_jackie_server(get_jackie_server())
    
    print("[SMAIT] Starting full AI pipeline...\n")
    config = get_config()
    if args.voice_only:
        config.voice_only = True
        print("[SMAIT] Voice-only mode — face detection disabled, listening immediately\n")
    else:
        config.voice_only = getattr(config, 'voice_only', False)
    hri = HRISystem(config)
    
    async def run():
        try:
            await hri.start()
            while hri._running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            await hri.stop()
            shutdown_jackie()
    
    asyncio.run(run())

if __name__ == "__main__":
    main()
