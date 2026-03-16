"""HRISystem: wires everything together and runs async loops."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Optional

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.connection.manager import ConnectionManager
from smait.sensors.audio_pipeline import AudioPipeline
from smait.sensors.video_pipeline import VideoPipeline
from smait.perception.face_tracker import FaceTracker
from smait.perception.lip_extractor import LipExtractor
from smait.perception.gaze import GazeEstimator
from smait.perception.engagement import EngagementDetector
from smait.perception.dolphin_separator import DolphinSeparator
from smait.perception.asr import ParakeetASR
from smait.perception.transcriber import Transcriber
from smait.perception.eou_detector import EOUDetector
from smait.dialogue.manager import DialogueManager
from smait.output.tts import TTSEngine
from smait.session.manager import SessionManager, SessionState
from smait.utils.data_logger import DataLogger
from smait.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

MAX_LOOP_RETRIES = 10
LOOP_COOLDOWN_S = 1.0


class HRISystem:
    """Top-level asyncio application.

    Init order:
    1. Load Config
    2. Create EventBus
    3. Start ConnectionManager (WebSocket server)
    4. Init AudioPipeline + VideoPipeline
    5. Init FaceTracker + LipExtractor + GazeEstimator + EngagementDetector
    6. Init DolphinSeparator (GPU)
    7. Init ParakeetASR (GPU)
    8. Init Transcriber
    9. Init EOUDetector (CPU)
    10. Init DialogueManager
    11. Init TTSEngine (GPU)
    12. Init SessionManager
    13. Init DataLogger
    14. Wire all event subscriptions
    15. Start async loops

    Crash recovery: auto-restart each loop on exception (max 10 retries, 1s cooldown).
    Graceful shutdown: Ctrl+C -> stop all -> save logs -> disconnect.
    """

    def __init__(self, config: Config, voice_only: bool = False) -> None:
        self._config = config
        self._voice_only = voice_only
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Core
        self.event_bus = EventBus()
        self.metrics = MetricsTracker()

        # Connection
        self.connection = ConnectionManager(config, self.event_bus)

        # Sensors
        self.audio_pipeline = AudioPipeline(config, self.event_bus)
        self.video_pipeline = VideoPipeline(config, self.event_bus)

        # Perception
        self.face_tracker = FaceTracker(config, self.event_bus) if not voice_only else None
        self.lip_extractor = LipExtractor(config, self.event_bus) if not voice_only else None
        self.gaze_estimator = GazeEstimator(config, self.event_bus) if not voice_only else None
        self.engagement_detector = EngagementDetector(config, self.event_bus) if not voice_only else None
        self.dolphin_separator = DolphinSeparator(config, self.event_bus)
        self.asr = ParakeetASR(config)
        self.transcriber = Transcriber(config, self.event_bus, self.asr)
        self.eou_detector = EOUDetector(config, self.event_bus)

        # Dialogue + Output
        self.dialogue = DialogueManager(config, self.event_bus)
        self.tts = TTSEngine(config, self.event_bus)

        # Session
        self.session = SessionManager(config, self.event_bus)

        # Logging
        self.data_logger = DataLogger(config, event_name="hfes")

    async def start(self) -> None:
        """Initialize all components and start the system."""
        logger.info("=== SMAIT HRI v3.0 Starting ===")
        logger.info("Config: voice_only=%s, debug=%s, show_video=%s",
                     self._voice_only, self._config.debug, self._config.show_video)

        # Start WebSocket server
        await self.connection.start()

        # Load ML models (heavy, GPU)
        logger.info("Loading ML models...")
        await self.audio_pipeline.init_model()

        if not self._voice_only:
            if self.gaze_estimator:
                await self.gaze_estimator.init_model()

        await self.dolphin_separator.init_model()
        await self.asr.init_model()
        await self.eou_detector.init_model()
        await self.tts.init_model()
        await self.dialogue.init()

        logger.info("All models loaded")

        # Wire event subscriptions
        self._wire_events()

        # Start async loops
        self._running = True
        self._tasks = [
            asyncio.create_task(self._resilient_loop("audio", self._audio_loop)),
            asyncio.create_task(self._resilient_loop("session_timeout", self._session_timeout_loop)),
        ]

        if not self._voice_only:
            self._tasks.append(
                asyncio.create_task(self._resilient_loop("video", self._video_loop))
            )

        logger.info("=== SMAIT HRI v3.0 Running ===")
        logger.info("Waiting for Jackie to connect on %s:%d...",
                     self._config.connection.host, self._config.connection.port)

    async def stop(self) -> None:
        """Gracefully shut down the system."""
        logger.info("Shutting down SMAIT HRI...")
        self._running = False

        # Cancel all loop tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # End current session
        session_log = self.data_logger.end_session()
        if session_log:
            logger.info("Final session score: %d/7", session_log.score)

        # Log metrics
        self.metrics.log_summary()

        # Stop connection
        await self.connection.stop()

        # Close resources
        if self.face_tracker:
            self.face_tracker.close()

        logger.info("=== SMAIT HRI v3.0 Stopped ===")

    async def run(self) -> None:
        """Start the system and run until interrupted."""
        await self.start()

        # Wait for shutdown signal
        stop_event = asyncio.Event()

        def signal_handler() -> None:
            logger.info("Shutdown signal received")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        await stop_event.wait()
        await self.stop()

    def _wire_events(self) -> None:
        """Connect all event subscriptions between modules."""

        # Connection → Audio/Video routing
        def on_incoming_data(data: object) -> None:
            if not isinstance(data, dict):
                return
            dtype = data.get("type")
            ts = data.get("timestamp", time.monotonic())
            if dtype == "cae":
                self.audio_pipeline.process_cae_audio(data["audio"], ts)
            elif dtype == "raw":
                self.audio_pipeline.process_raw_audio(data["audio"], ts)

        self.event_bus.subscribe(EventType.SPEECH_DETECTED, on_incoming_data)

        # Connection → Video routing (FACE_UPDATED carries JPEG from Jackie)
        def on_video_frame(data: object) -> None:
            if not isinstance(data, dict):
                return
            if data.get("type") == "video" and "jpeg" in data:
                self.video_pipeline.process_jpeg(
                    data["jpeg"], data.get("timestamp", time.monotonic())
                )

        self.event_bus.subscribe(EventType.FACE_UPDATED, on_video_frame)

        # Speech segment → Separation → ASR
        async def on_speech_segment(segment: object) -> None:
            from smait.sensors.audio_pipeline import SpeechSegment
            if not isinstance(segment, SpeechSegment):
                return

            # Engagement gating: only process speech when engaged/conversing
            # In voice_only mode, skip gating (no vision to detect engagement)
            if not self._voice_only and self.session.state not in (
                SessionState.ENGAGED, SessionState.CONVERSING
            ):
                logger.debug("Ignoring speech segment — session state: %s",
                             self.session.state.name)
                return

            self.metrics.start_timer("separation")

            # Get lip frames for target face
            lip_frames = []
            if self.lip_extractor and self.session.target_track_id is not None:
                lip_frames = self.lip_extractor.get_lip_frames(
                    self.session.target_track_id,
                    segment.start_time,
                    segment.end_time,
                )

            # Fallback: if initial lip frame lookup found nothing, try recent frames
            if not lip_frames and self.lip_extractor:
                lip_frames = self.lip_extractor.get_recent_frames(
                    self.session.target_track_id, count=25
                )

            # Run separation — prefer raw 4-channel audio when available
            # Raw multichannel preserves spatial information from 4-mic array.
            # Dolphin downmixes to mono internally but having the original
            # channels gives better signal than CAE's single beamformed output
            # when the beam is pointed at the wrong speaker.
            if segment.raw_audio is not None and self._config.separation.use_multichannel:
                separation = await self.dolphin_separator.separate(
                    segment.raw_audio,
                    lip_frames,
                    channels=self._config.audio.channels_raw,
                )
            else:
                separation = await self.dolphin_separator.separate(
                    segment.cae_audio,
                    lip_frames,
                    channels=1,
                )

            sep_ms = self.metrics.stop_timer("separation")
            self.metrics.record("separation_confidence", separation.separation_confidence * 100)

            # Save audio for debugging
            self.data_logger.save_audio_wav(
                segment.cae_audio, f"turn_{time.monotonic():.0f}_cae.wav",
                sample_rate=self._config.audio.sample_rate,
            )
            self.data_logger.save_audio_wav(
                separation.separated_audio, f"turn_{time.monotonic():.0f}_separated.wav",
                sample_rate=self._config.audio.sample_rate,
            )

            self._event_bus_emit_separated(separation, segment)

        self.event_bus.subscribe(EventType.SPEECH_SEGMENT, on_speech_segment)

        # Separated audio → ASR
        async def on_speech_separated(data: object) -> None:
            if not isinstance(data, dict):
                return

            self.metrics.start_timer("asr")

            from smait.perception.dolphin_separator import SeparationResult
            separation = data.get("separation")
            if separation is None:
                return

            result = await self.transcriber.process_separated_audio(
                separation,
                start_time=data.get("start_time", 0),
                end_time=data.get("end_time", 0),
            )

            asr_ms = self.metrics.stop_timer("asr")

            if result:
                # The speech segment was already cut by VAD silence detection,
                # so this is a complete utterance. Emit END_OF_TURN directly.
                now = time.monotonic()
                logger.info("Emitting END_OF_TURN for: '%s'", result.text)
                self.event_bus.emit(EventType.END_OF_TURN, {
                    "text": result.text,
                    "confidence": result.confidence,
                    "reason": "vad_segment_complete",
                    "timestamp": now,
                    "silence_ms": 0,
                })

                # Update turn log
                turn = self.data_logger.start_turn()
                turn.user_text = result.text
                turn.asr_confidence = result.confidence
                turn.asr_latency_ms = result.latency_ms
                if separation:
                    turn.dolphin_confidence = separation.separation_confidence
                    turn.separation_snr = separation.separation_confidence

        self.event_bus.subscribe(EventType.SPEECH_SEPARATED, on_speech_separated)

        # End of turn → Dialogue
        async def on_end_of_turn(data: object) -> None:
            if not isinstance(data, dict):
                return

            text = data.get("text", "")
            if not text:
                return

            # Engagement gating: only respond when engaged/conversing
            # In voice_only mode, skip gating (no vision to detect engagement)
            if not self._voice_only and self.session.state not in (
                SessionState.ENGAGED, SessionState.CONVERSING
            ):
                logger.debug("Ignoring end-of-turn — session state: %s",
                             self.session.state.name)
                return

            self.metrics.start_timer("dialogue")

            # Send user transcript to Jackie
            asyncio.create_task(self.connection.send_transcript(text, "user"))
            asyncio.create_task(self.connection.send_state("engaged", "thinking"))

            # Stream LLM response → TTS
            full_response = ""
            async def response_gen():
                nonlocal full_response
                async for chunk in self.dialogue.ask_streaming(text):
                    full_response += chunk
                    yield chunk

            await self.tts.speak_streaming(response_gen())

            llm_ms = self.metrics.stop_timer("dialogue")

            # Send robot response to Jackie
            if full_response:
                asyncio.create_task(self.connection.send_transcript(full_response, "robot"))
                asyncio.create_task(self.connection.send_state("engaged", "listening"))

                # Update turn log
                if self.data_logger._current_turn:
                    self.data_logger._current_turn.robot_text = full_response
                    self.data_logger._current_turn.llm_latency_ms = llm_ms or 0
                    self.data_logger.end_turn()

        self.event_bus.subscribe(EventType.END_OF_TURN, on_end_of_turn)

        # VAD probabilities → EOU detector (primary turn-taking path)
        def on_vad_prob(data: object) -> None:
            if isinstance(data, dict):
                self.eou_detector.feed_vad_prob(
                    data["speech_prob"],
                    data["n_samples"],
                    data["timestamp"],
                )

        self.event_bus.subscribe(EventType.VAD_PROB, on_vad_prob)

        # CAE status check (Issue #1)
        def on_cae_status(data: object) -> None:
            if isinstance(data, dict):
                self.data_logger.set_cae_status(data)

        self.event_bus.subscribe(EventType.CAE_STATUS, on_cae_status)

        # DOA tracking
        def on_doa(data: object) -> None:
            if isinstance(data, dict):
                self.data_logger.add_doa_angle(data.get("angle", 0))

        self.event_bus.subscribe(EventType.DOA_UPDATE, on_doa)

        # Session events
        async def on_session_start(data: object) -> None:
            if isinstance(data, dict):
                sid = data.get("session_id", "unknown")
                self.data_logger.start_session(sid, data)

                # Proactive greeting
                greeting = data.get("greeting", "")
                if greeting:
                    await self.tts.speak(greeting)
                    await self.connection.send_transcript(greeting, "robot")

        self.event_bus.subscribe(EventType.SESSION_START, on_session_start)

        async def on_session_end_log(data: object) -> None:
            clean = isinstance(data, dict) and data.get("reason") == "goodbye_detected"
            self.data_logger.end_session(clean_farewell=clean)
            self.dialogue.clear_history()
            self.eou_detector.reset()
            if self.engagement_detector:
                self.engagement_detector.reset()

        self.event_bus.subscribe(EventType.SESSION_END, on_session_end_log)

    def _event_bus_emit_separated(self, separation, segment) -> None:
        """Helper to emit SPEECH_SEPARATED with combined data."""
        self.event_bus.emit(EventType.SPEECH_SEPARATED, {
            "separation": separation,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
        })

    # --- Async Loops ---

    async def _resilient_loop(self, name: str, loop_fn) -> None:
        """Run a loop with auto-restart on exception."""
        retries = 0
        while self._running and retries < MAX_LOOP_RETRIES:
            try:
                await loop_fn()
            except asyncio.CancelledError:
                break
            except Exception:
                retries += 1
                logger.exception("Loop '%s' crashed (retry %d/%d)",
                                 name, retries, MAX_LOOP_RETRIES)
                if retries < MAX_LOOP_RETRIES:
                    await asyncio.sleep(LOOP_COOLDOWN_S)

        if retries >= MAX_LOOP_RETRIES:
            logger.error("Loop '%s' exceeded max retries, stopping", name)

    async def _audio_loop(self) -> None:
        """Process incoming audio: CAE → VAD → speech segments.

        The actual audio processing is event-driven via SPEECH_DETECTED events
        from ConnectionManager. This loop handles EOU silence checking.
        """
        while self._running:
            # Check for silence-based EOU
            if self.session.state == SessionState.CONVERSING:
                self.eou_detector.on_silence(time.monotonic())
            await asyncio.sleep(0.05)  # 50ms check interval

    async def _video_loop(self) -> None:
        """Process video frames: face tracking → gaze → engagement → lip ROI.

        Frames arrive via FACE_UPDATED events from ConnectionManager.
        This loop processes them through the vision pipeline.
        When show_video is enabled, displays annotated frames via OpenCV.
        """
        cv2 = None
        if self._config.show_video:
            try:
                import cv2 as _cv2
                cv2 = _cv2
                logger.info("Video display enabled — will show SMAIT Vision window")
            except ImportError:
                logger.warning("show_video enabled but cv2 not available")

        try:
            while self._running:
                # Wait for connection
                if not self.connection.connected:
                    await asyncio.sleep(0.5)
                    continue

                # Process latest video frame
                frame_data = self.video_pipeline.latest_frame
                if frame_data is None:
                    await asyncio.sleep(0.033)  # ~30 FPS polling
                    continue

                timestamp = frame_data.timestamp

                # Face tracking
                tracks = self.face_tracker.process_frame(frame_data.image, timestamp)

                # Gaze estimation + Lip extraction for each tracked face
                gaze_results = {}
                for track in tracks:
                    gaze = self.gaze_estimator.estimate(frame_data.image, track, timestamp)
                    gaze_results[track.track_id] = gaze

                    if self.lip_extractor:
                        self.lip_extractor.extract(frame_data.image, track, timestamp)

                # Engagement detection (pass frame width for DOA-to-pixel mapping)
                if self.engagement_detector:
                    frame_w = frame_data.image.shape[1]
                    self.engagement_detector.update(tracks, gaze_results, timestamp, frame_w)

                # Display annotated video on lab PC
                if cv2 is not None:
                    display = frame_data.image.copy()
                    for track in tracks:
                        x, y, w, h = track.bbox
                        gaze_r = gaze_results.get(track.track_id)
                        looking = gaze_r and gaze_r.is_looking_at_robot
                        color = (0, 255, 0) if looking else (255, 255, 0)
                        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                        label = f"#{track.track_id}"
                        if looking:
                            label += " LOOKING"
                        cv2.putText(display, label,
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.imshow("SMAIT Vision", display)
                    cv2.waitKey(1)

                await asyncio.sleep(0.033)  # ~30 FPS
        finally:
            if cv2 is not None:
                cv2.destroyAllWindows()

    async def _session_timeout_loop(self) -> None:
        """Periodically check for session timeouts."""
        while self._running:
            await self.session.check_timeouts()
            await asyncio.sleep(1.0)
