"""Individual component demo scripts.

Usage: python -m smait.demo <component>

Components:
  connection  — test Jackie WebSocket connectivity + frame reception
  audio       — test VAD with live audio, show probability bar
  video       — test face tracking + gaze + lip ROI with camera display
  separation  — test Dolphin with live audio + video
  asr         — test Parakeet transcription on clean audio
  tts         — test Kokoro TTS with text input
  dialogue    — test LLM dialogue via text input
  eou         — test end-of-utterance detection with typed sentences
  full        — run complete HRI system
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smait.demo")


async def demo_connection() -> None:
    """Test Jackie WebSocket connectivity + frame reception."""
    from smait.core.config import Config
    from smait.core.events import EventBus, EventType
    from smait.connection.manager import ConnectionManager

    config = Config()
    bus = EventBus()

    frame_counts = {"audio_cae": 0, "audio_raw": 0, "video": 0, "doa": 0}
    start_time = time.monotonic()

    def on_audio(data):
        if isinstance(data, dict):
            if data.get("type") == "cae":
                frame_counts["audio_cae"] += 1
            elif data.get("type") == "raw":
                frame_counts["audio_raw"] += 1

    def on_video(data):
        frame_counts["video"] += 1

    def on_doa(data):
        frame_counts["doa"] += 1
        if isinstance(data, dict):
            logger.info("DOA: angle=%d beam=%d", data.get("angle", 0), data.get("beam", 0))

    bus.subscribe(EventType.SPEECH_DETECTED, on_audio)
    bus.subscribe(EventType.FACE_UPDATED, on_video)
    bus.subscribe(EventType.DOA_UPDATE, on_doa)
    bus.subscribe(EventType.CAE_STATUS, lambda d: logger.info("CAE status: %s", d))

    conn = ConnectionManager(config, bus)
    await conn.start()

    logger.info("Waiting for Jackie to connect on %s:%d...", config.connection.host, config.connection.port)
    await conn.wait_for_connection()
    logger.info("Jackie connected!")

    try:
        while True:
            await asyncio.sleep(5.0)
            elapsed = time.monotonic() - start_time
            logger.info(
                "Frames received (%.0fs): cae=%d raw=%d video=%d doa=%d",
                elapsed, frame_counts["audio_cae"], frame_counts["audio_raw"],
                frame_counts["video"], frame_counts["doa"],
            )
    except asyncio.CancelledError:
        pass
    finally:
        await conn.stop()


async def demo_audio() -> None:
    """Test VAD with live audio from Jackie."""
    from smait.core.config import Config
    from smait.core.events import EventBus, EventType
    from smait.connection.manager import ConnectionManager
    from smait.sensors.audio_pipeline import AudioPipeline

    config = Config()
    bus = EventBus()

    audio = AudioPipeline(config, bus)
    await audio.init_model()

    segment_count = 0

    def on_segment(seg):
        nonlocal segment_count
        segment_count += 1
        logger.info("Speech segment #%d: duration=%.2fs, samples=%d",
                     segment_count, seg.duration, len(seg.cae_audio))

    bus.subscribe(EventType.SPEECH_SEGMENT, on_segment)

    def on_audio_data(data):
        if isinstance(data, dict):
            ts = data.get("timestamp", time.monotonic())
            if data.get("type") == "cae":
                audio.process_cae_audio(data["audio"], ts)
            elif data.get("type") == "raw":
                audio.process_raw_audio(data["audio"], ts)

    bus.subscribe(EventType.SPEECH_DETECTED, on_audio_data)

    conn = ConnectionManager(config, bus)
    await conn.start()

    logger.info("Waiting for Jackie...")
    await conn.wait_for_connection()
    logger.info("Jackie connected! Listening for speech...")

    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    finally:
        await conn.stop()


async def demo_video() -> None:
    """Test face tracking + gaze + lip ROI with camera display."""
    import cv2
    from smait.core.config import Config
    from smait.core.events import EventBus, EventType
    from smait.connection.manager import ConnectionManager
    from smait.sensors.video_pipeline import VideoPipeline
    from smait.perception.face_tracker import FaceTracker
    from smait.perception.gaze import GazeEstimator
    from smait.perception.lip_extractor import LipExtractor

    config = Config()
    bus = EventBus()

    video = VideoPipeline(config, bus)
    tracker = FaceTracker(config, bus)
    gaze = GazeEstimator(config, bus)
    await gaze.init_model()
    lip = LipExtractor(config, bus)

    def on_video_frame(data):
        if not isinstance(data, dict) or data.get("type") != "video":
            return

        frame = video.process_jpeg(data["jpeg"], data["timestamp"])
        if frame is None:
            return

        tracks = tracker.process_frame(frame.image, frame.timestamp)

        display = frame.image.copy()
        for track in tracks:
            x, y, w, h = track.bbox
            color = (0, 255, 0) if track.is_target else (255, 255, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            gaze_result = gaze.estimate(frame.image, track, frame.timestamp)
            looking = "LOOKING" if gaze_result.is_looking_at_robot else ""
            cv2.putText(display, f"#{track.track_id} {looking}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            roi = lip.extract(frame.image, track, frame.timestamp)
            if roi:
                cv2.imshow(f"Lip ROI #{track.track_id}", roi.image)

        cv2.imshow("SMAIT Vision", display)
        cv2.waitKey(1)

    bus.subscribe(EventType.FACE_UPDATED, on_video_frame)

    conn = ConnectionManager(config, bus)
    await conn.start()

    logger.info("Waiting for Jackie...")
    await conn.wait_for_connection()
    logger.info("Jackie connected! Showing video...")

    try:
        while True:
            await asyncio.sleep(0.033)
    except asyncio.CancelledError:
        pass
    finally:
        await conn.stop()
        cv2.destroyAllWindows()
        tracker.close()


async def demo_asr() -> None:
    """Test Parakeet transcription on clean audio file or microphone."""
    import numpy as np
    from smait.core.config import Config
    from smait.perception.asr import ParakeetASR

    config = Config()
    asr = ParakeetASR(config)
    await asr.init_model()

    if not asr.available:
        logger.error("ASR model not available")
        return

    logger.info("ASR ready. Enter audio file paths or 'mic' for microphone input.")
    logger.info("Type 'quit' to exit.")

    while True:
        try:
            path = input("\nAudio file path (or 'quit'): ").strip()
        except EOFError:
            break

        if path.lower() == "quit":
            break

        if path.lower() == "mic":
            try:
                import sounddevice as sd
                logger.info("Recording 5 seconds...")
                audio = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype="float32")
                sd.wait()
                audio = audio.squeeze()
            except ImportError:
                logger.error("sounddevice not installed")
                continue
        else:
            try:
                import wave
                with wave.open(path) as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception:
                logger.error("Failed to read %s", path)
                continue

        result = asr.transcribe(audio)
        if result:
            logger.info("Text: '%s'", result.text)
            logger.info("Confidence: %.2f, Latency: %.1fms", result.confidence, result.latency_ms)


async def demo_tts() -> None:
    """Test Kokoro TTS with text input."""
    import numpy as np
    from smait.core.config import Config
    from smait.core.events import EventBus
    from smait.output.tts import TTSEngine

    config = Config()
    bus = EventBus()
    tts = TTSEngine(config, bus)
    await tts.init_model()

    if not tts.available:
        logger.error("TTS model not available")
        return

    logger.info("TTS ready. Type text to synthesize. Type 'quit' to exit.")

    while True:
        try:
            text = input("\nText: ").strip()
        except EOFError:
            break

        if text.lower() == "quit":
            break

        pcm = await tts.synthesize(text)
        if pcm:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            logger.info("Generated %d samples (%.2fs at 24kHz)", len(audio), len(audio) / 24000)
            try:
                import sounddevice as sd
                sd.play(audio, samplerate=24000)
                sd.wait()
            except ImportError:
                logger.info("sounddevice not installed — cannot play audio")


async def demo_dialogue() -> None:
    """Test LLM dialogue via text input."""
    from smait.core.config import Config
    from smait.core.events import EventBus
    from smait.dialogue.manager import DialogueManager

    config = Config()
    bus = EventBus()
    dm = DialogueManager(config, bus)
    await dm.init()

    logger.info("Dialogue ready. Type messages as the user. Type 'quit' to exit.")

    while True:
        try:
            user = input("\nYou: ").strip()
        except EOFError:
            break

        if user.lower() == "quit":
            break

        response = await dm.ask(user)
        print(f"Jackie: {response.text}")
        print(f"  (model={response.model_used}, latency={response.latency_ms:.0f}ms, farewell={response.is_farewell})")


async def demo_eou() -> None:
    """Test end-of-utterance detection with typed sentences."""
    from smait.core.config import Config
    from smait.core.events import EventBus, EventType
    from smait.perception.eou_detector import EOUDetector

    config = Config()
    bus = EventBus()

    def on_eou(data):
        logger.info("END OF TURN: %s", data)

    bus.subscribe(EventType.END_OF_TURN, on_eou)

    eou = EOUDetector(config, bus)
    await eou.init_model()

    logger.info("EOU detector ready. Type partial sentences. Type 'quit' to exit.")
    logger.info("Use '--' on a new line to simulate silence.")

    while True:
        try:
            text = input("\nText: ").strip()
        except EOFError:
            break

        if text.lower() == "quit":
            break

        if text == "--":
            eou.on_silence(time.monotonic())
        else:
            eou.update_transcript(text, time.monotonic())
            prob = eou.predict(text)
            print(f"  P(end_of_turn) = {prob:.3f}")


async def demo_separation() -> None:
    """Test Dolphin AV-TSE with audio + video."""
    import numpy as np
    from smait.core.config import Config
    from smait.core.events import EventBus
    from smait.perception.dolphin_separator import DolphinSeparator

    config = Config()
    bus = EventBus()
    separator = DolphinSeparator(config, bus)
    await separator.init_model()

    logger.info("Dolphin separator available: %s", separator.available)

    # Generate test audio (sine wave + noise)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    clean = np.sin(2 * np.pi * 440 * t) * 0.5
    noise = np.random.randn(len(t)) * 0.3
    mixed = (clean + noise).astype(np.float32)
    mixed_int16 = (mixed * 32767).clip(-32768, 32767).astype(np.int16)

    result = await separator.separate(mixed_int16, [], channels=1)
    logger.info("Separation result: conf=%.2f, latency=%.1fms, multichannel=%s",
                 result.separation_confidence, result.latency_ms, result.used_multichannel)


async def demo_full() -> None:
    """Run complete HRI system."""
    from smait.core.config import Config
    from smait.main import HRISystem

    config = Config()
    config.debug = True
    system = HRISystem(config)
    await system.run()


DEMOS = {
    "connection": demo_connection,
    "audio": demo_audio,
    "video": demo_video,
    "separation": demo_separation,
    "asr": demo_asr,
    "tts": demo_tts,
    "dialogue": demo_dialogue,
    "eou": demo_eou,
    "full": demo_full,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in DEMOS:
        print(__doc__)
        print("Available components:", ", ".join(sorted(DEMOS.keys())))
        sys.exit(1)

    component = sys.argv[1]
    logger.info("Running demo: %s", component)

    try:
        asyncio.run(DEMOS[component]())
    except KeyboardInterrupt:
        logger.info("Demo interrupted")


if __name__ == "__main__":
    main()
