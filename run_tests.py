#!/usr/bin/env python3
"""
SMAIT HRI Test Runner
=====================
Runs the full system with the TestHarness attached, records all metrics,
and generates a report for the Masters project report.

Usage:
    python run_tests.py                      # 15 sessions, Jackie mode
    python run_tests.py --sessions 20        # 20 sessions
    python run_tests.py --voice-only         # no camera (audio-only test)
    python run_tests.py --test-name asr_test # custom test name
    python run_tests.py --wer                # ASR accuracy mode (prompts for reference text)
"""

import argparse
import asyncio
import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from smait.core.config import get_config
from smait.main import HRISystem, init_jackie_sources, set_jackie_server
from smait.utils.test_harness import TestHarness


def main():
    parser = argparse.ArgumentParser(description="SMAIT HRI Test Runner")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sessions", type=int, default=15, help="Number of test sessions")
    parser.add_argument("--voice-only", action="store_true", help="Skip face detection")
    parser.add_argument("--test-name", default="smait_test", help="Label for output files")
    parser.add_argument("--output-dir", default="test_results", help="Where to save results")
    parser.add_argument("--wer", action="store_true", help="Prompt for reference text after each turn (WER mode)")
    args = parser.parse_args()

    config = get_config()
    if args.voice_only:
        config.voice_only = True
        print("[TEST] Voice-only mode — no face detection")

    # Init Jackie sources
    from smait.sensors.network_source import JackieWebSocketServer
    server = JackieWebSocketServer(host=args.host, port=args.port)
    init_jackie_sources(server)
    set_jackie_server(server)

    hri = HRISystem(config)

    # Attach harness
    harness = TestHarness(
        hri_system=hri,
        output_dir=args.output_dir,
        num_sessions=args.sessions,
        test_name=args.test_name
    )

    # Hook session end signals from main.py
    # Patch main.py's session-end paths to call harness.mark_session_end()
    original_dialogue_reset = hri.dialogue.reset_session if hri.dialogue else None

    async def patched_main():
        # Intercept session ends by monitoring state changes
        # We hook into HRISystem._timeout_loop and _on_transcript
        # by wrapping the send_state calls

        from smait.sensors import network_source as ns
        orig_send_state = None

        jackie_server = ns._jackie_server
        if jackie_server:
            orig_send_state = jackie_server.send_state

            async def hooked_send_state(state: str):
                if state == "idle":
                    # Determine reason
                    reason = "timeout"  # default
                    harness.mark_session_end(reason)
                elif state == "engaged" and harness._current_session is None:
                    pass  # session start handled via verifier hook
                if orig_send_state:
                    await orig_send_state(state)

            jackie_server.send_state = hooked_send_state

        harness.attach()
        print(f"\n[TEST] Test runner ready — recording {args.sessions} sessions")
        print(f"[TEST] Walk up to Jackie to begin. Results → {args.output_dir}/")
        print("[TEST] Press Ctrl+C when done to generate report\n")

        await hri.start()

    def on_shutdown(signum, frame):
        print("\n\n[TEST] Shutting down — generating report...")
        harness.report()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    try:
        asyncio.run(patched_main())
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted — generating report...")
        harness.report()


if __name__ == "__main__":
    main()
