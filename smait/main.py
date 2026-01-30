"""
SMAIT HRI System v2.0 - Main Entry Point (Integrated)

Features:
- Behavior Tree architecture for parallel behaviors
- Semantic VAD for early turn prediction
- Audio-visual synchronized verification
"""

import asyncio
import signal
import threading
import time
from typing import Optional
import cv2

from smait.core.config import get_config, set_config, Config, DeploymentMode
from smait.core.events import (
    TranscriptResult, SessionState, VerifyResult, VerifyOutput,
    DialogueResponse
)
from smait.sensors.audio_pipeline import AudioPipeline
from smait.sensors.sources import create_video_source
from smait.perception.transcriber import Transcriber
from smait.perception.verifier import SpeakerVerifier
from smait.perception.semantic_vad import SemanticVAD, TurnTakingManager
from smait.dialogue.manager import DialogueManager
from smait.output.tts import EdgeTTSEngine, PiperTTSEngine, TTSPlayer

# Behavior tree imports
try:
    from smait.behavior.tree import HRIBehaviorTree, get_blackboard
    BT_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Behavior trees not available: {e}")
    print("[WARNING] Install py_trees: pip install py_trees")
    BT_AVAILABLE = False


class HRISystem:
    """
    Main Human-Robot Interaction System with Behavior Trees and Semantic VAD.
    
    Architecture:
    - Behavior Tree manages high-level interaction flow
    - Semantic VAD predicts turn completion for faster responses
    - Audio pipeline provides speech segments
    - Verifier ensures speech matches visible speaker
    
    Flow:
        Audio → VAD → ASR → Semantic VAD → BT → Verify → Dialogue
                              ↓ (early)
                         Prepare Response
    """
    
    def __init__(self, config: Optional[Config] = None):
        if config:
            set_config(config)
        self.config = get_config()
        
        # Core components
        self.audio_pipeline: Optional[AudioPipeline] = None
        self.transcriber: Optional[Transcriber] = None
        self.verifier: Optional[SpeakerVerifier] = None
        self.dialogue: Optional[DialogueManager] = None
        
        # New components
        self.semantic_vad: Optional[SemanticVAD] = None
        self.turn_manager: Optional[TurnTakingManager] = None
        self.behavior_tree: Optional[HRIBehaviorTree] = None

        # TTS
        self.tts_engine: Optional[EdgeTTSEngine] = None
        self.tts_player: Optional[TTSPlayer] = None
        
        # Video
        self._video_thread: Optional[threading.Thread] = None
        
        # State
        self._running = False
        self._last_activity: Optional[float] = None
        self._preparing_response = False
        self._early_response_text: Optional[str] = None
        
        # Tasks
        self._main_task: Optional[asyncio.Task] = None
        self._timeout_task: Optional[asyncio.Task] = None
        self._bt_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._response_times = []
        self._early_predictions = 0
        self._total_turns = 0
    
    async def start(self):
        """Initialize and start the HRI system"""
        print("=" * 60)
        print("SMAIT HRI System v2.0 (Integrated)")
        print("=" * 60)
        print(f"Mode: {self.config.mode.value}")
        print(f"ASR: {self.config.asr.backend.value}")
        print(f"LLM: {self.config.dialogue.llm_backend.value} ({self.config.dialogue.model_name})")
        print(f"Behavior Trees: {'Enabled' if BT_AVAILABLE else 'Disabled'}")
        print(f"Semantic VAD: Enabled")
        print(f"Debug: {self.config.debug}")
        print("=" * 60)
        print()
        
        # Initialize components
        print("[SYSTEM] Initializing components...")
        
        # Audio pipeline
        self.audio_pipeline = AudioPipeline()
        self.audio_pipeline.set_vad_callback(self._on_vad_change)
        
        # Transcriber
        self.transcriber = Transcriber(self.audio_pipeline)
        self.transcriber.set_final_callback(self._on_transcript)
        
        # Speaker Verifier
        self.verifier = SpeakerVerifier()
        
        # Connect audio pipeline to verifier for ASD synchronization
        self.audio_pipeline.set_speech_start_callback(self.verifier.mark_speech_start)
        self.audio_pipeline.set_speech_end_callback(self.verifier.mark_speech_end)
        
        # Dialogue manager
        self.dialogue = DialogueManager()
        self.dialogue.set_response_callback(self._on_response)
        
        # === Semantic VAD for turn prediction ===
        self.semantic_vad = SemanticVAD(use_llm=False)  # Start with pattern-based only
        self.turn_manager = TurnTakingManager(
            on_turn_complete=self._on_turn_predicted_complete,
            on_early_prediction=self._on_early_turn_prediction
        )
        print("[SEMANTIC-VAD] Turn prediction enabled")
        
        # === TTS Engine (Piper = fast local, Edge = natural cloud) ===
        try:
            self.tts_engine = PiperTTSEngine(voice="amy", use_cuda=False)
            if self.tts_engine._voice is None:
                raise RuntimeError("Piper voice not loaded")
            print("[TTS] Piper TTS ready (local, fast)")
        except Exception as e:
            print(f"[TTS] Piper failed ({e}), falling back to Edge TTS")
            self.tts_engine = EdgeTTSEngine(voice="en-us-aria")
            print("[TTS] Edge TTS ready (cloud)")
        self.tts_player = TTSPlayer(play_locally=True)

        # === Behavior Tree ===
        if BT_AVAILABLE:
            self.behavior_tree = HRIBehaviorTree()
            self.behavior_tree.setup(
                audio_pipeline=self.audio_pipeline,
                transcriber=self.transcriber,
                verifier=self.verifier,
                dialogue=self.dialogue
            )
        
        # Start components
        print("[SYSTEM] Starting components...")
        self.audio_pipeline.start()
        await self.transcriber.start()
        
        # Start video thread
        self._running = True
        self._video_thread = threading.Thread(
            target=self._video_loop,
            daemon=True,
            name="VideoCapture"
        )
        self._video_thread.start()
        
        # Start processing tasks
        self._main_task = asyncio.create_task(self._main_loop())
        self._timeout_task = asyncio.create_task(self._timeout_loop())
        
        # Start BT tick loop if available
        if BT_AVAILABLE and self.behavior_tree:
            self._bt_task = asyncio.create_task(self._bt_loop())
        
        print("[SYSTEM] Ready! Speak to interact.\n")
    
    async def stop(self):
        """Stop the HRI system"""
        print("\n[SYSTEM] Shutting down...")
        
        self._running = False
        
        # Print metrics
        if self._total_turns > 0:
            print(f"\n[METRICS] Total turns: {self._total_turns}")
            print(f"[METRICS] Early predictions: {self._early_predictions}")
            if self._response_times:
                avg_time = sum(self._response_times) / len(self._response_times)
                print(f"[METRICS] Avg response time: {avg_time:.0f}ms")
        
        # Cancel tasks
        for task in [self._main_task, self._timeout_task, self._bt_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop video
        if self._video_thread:
            self._video_thread.join(timeout=2.0)
        
        # Stop components
        if self.transcriber:
            await self.transcriber.stop()
        if self.audio_pipeline:
            self.audio_pipeline.stop()
        if self.verifier:
            self.verifier.cleanup()
        if self.behavior_tree:
            self.behavior_tree.shutdown()
        
        cv2.destroyAllWindows()
        
        print("[SYSTEM] Shutdown complete")
    
    def _video_loop(self):
        """Video capture and processing loop"""
        from smait.sensors.sources import CameraSource
        
        try:
            camera = CameraSource(
                device=self.config.vision.camera_index,
                width=self.config.vision.frame_width,
                height=self.config.vision.frame_height
            )
            camera.start()
        except Exception as e:
            print(f"[VIDEO] Failed to start camera: {e}")
            return
        
        if self.config.show_video:
            cv2.namedWindow("SMAIT HRI v2.0", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        try:
            while self._running:
                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame through verifier
                result = self.verifier.process_frame(frame)
                
                # Update BT blackboard with current state
                if BT_AVAILABLE and self.behavior_tree:
                    bb = get_blackboard()
                    bb.current_faces = result['faces']
                    bb.current_asd_results = result['asd_results']
                    bb.primary_speaker_id = result['primary_speaker_id']
                
                # Draw additional info
                if result['frame'] is not None:
                    self._draw_overlay(result['frame'])
                
                # Show video
                if self.config.show_video and result['frame'] is not None:
                    cv2.imshow("SMAIT HRI v2.0", result['frame'])
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        self._running = False
                        break
                
                frame_count += 1
                
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    
    def _draw_overlay(self, frame):
        """Draw system status overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semantic VAD status
        if self._preparing_response:
            cv2.putText(
                frame,
                "Preparing response...",
                (w - 250, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        # BT status
        if BT_AVAILABLE and self.behavior_tree:
            state = self.behavior_tree.session_state.name
            cv2.putText(
                frame,
                f"BT: {state}",
                (w - 200, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
    
    async def _bt_loop(self):
        """Behavior tree tick loop"""
        tick_rate = 30  # Hz
        tick_interval = 1.0 / tick_rate
        
        while self._running:
            try:
                if self.behavior_tree:
                    self.behavior_tree.tick()
                    
                    # Check if BT has a response ready
                    response = self.behavior_tree.get_response()
                    if response:
                        print(f"[BT-ROBOT] {response.text}")
                
                await asyncio.sleep(tick_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.debug:
                    print(f"[BT] Error: {e}")
    
    async def _main_loop(self):
        """Main transcript processing loop"""
        try:
            async for transcript in self.transcriber.transcripts():
                if not self._running:
                    break
                
                await self._process_transcript(transcript)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[SYSTEM] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_transcript(self, transcript: TranscriptResult):
        """Process a transcription result with Semantic VAD"""
        text = transcript.text.strip()
        
        if not text:
            return
        
        start_time = time.time()
        self._total_turns += 1
        
        # === Semantic VAD: Analyze turn completion ===
        # Note: This is for final transcripts, so we always process them
        # The "text_changing" check is only relevant for streaming partial transcripts
        prediction = self.semantic_vad.predict(text)
        
        # Verify speech came from target user
        verification = self.verifier.verify_speech(transcript)
        
        # Feed to behavior tree if available
        if BT_AVAILABLE and self.behavior_tree:
            self.behavior_tree.feed_transcript(transcript)
        
        if verification.result == VerifyResult.ACCEPT:
            # Print semantic VAD analysis
            if self.config.debug:
                print(f"\n[TURN] \"{text}\"")
                print(f"       Semantic: complete={prediction.is_complete}, conf={prediction.confidence:.2f} ({prediction.reason})")
                print(f"       Visual: speaker={verification.face_id}, conf={verification.confidence:.2f}")
            else:
                print(f"\n[USER] \"{text}\"")
            
            # Check if we have an early-prepared response
            if self._early_response_text == text and self._preparing_response:
                if self.config.debug:
                    print("       [Early prediction used!]")
                self._early_predictions += 1
            
            # Generate response
            response = await self.dialogue.ask_async(text, transcript.confidence)

            response_time = (time.time() - start_time) * 1000
            self._response_times.append(response_time)

            print(f"[ROBOT] {response.text}")
            if self.config.debug:
                print(f"        (latency: {response_time:.0f}ms)")

            # Synthesize and play TTS
            if self.tts_engine and self.tts_player:
                try:
                    tts_result = await self.tts_engine.synthesize(response.text)
                    if self.config.debug:
                        print(f"        (TTS: {tts_result.latency_ms:.0f}ms)")
                    # Play in background to not block
                    asyncio.create_task(self.tts_player.play_async(tts_result))
                except Exception as e:
                    print(f"[TTS] Error: {e}")

            # Check if session should end (user said goodbye)
            if self.verifier.check_pending_session_end():
                self.dialogue.reset_session()
                if BT_AVAILABLE and self.behavior_tree:
                    bb = get_blackboard()
                    bb.session_state = SessionState.IDLE
                    bb.target_user_id = None

            self._last_activity = time.time()
            self._preparing_response = False
            self._early_response_text = None
            
        elif verification.result == VerifyResult.REJECT:
            # Only print rejection if debug AND it's a real rejection (not just noise)
            if self.config.debug and len(text) > 5:
                print(f"[REJECT] \"{text[:40]}\" - {verification.reason}")
            self._preparing_response = False
            self._early_response_text = None
        
        elif verification.result == VerifyResult.NO_FACE:
            if self.config.debug and len(text) > 5:
                print(f"[NO_FACE] \"{text[:40]}\" - no face visible")
        
        # Reset turn manager
        self.turn_manager.confirm_end_of_turn(text)
    
    def _on_early_turn_prediction(self, text: str, confidence: float):
        """
        Called when Semantic VAD predicts turn is complete BEFORE silence timeout.
        This allows us to start preparing the response early.
        """
        if confidence > 0.8 and not self._preparing_response:
            self._preparing_response = True
            self._early_response_text = text
            
            if self.config.debug:
                print(f"[SEMANTIC-VAD] Early turn prediction! Starting response prep...")
            
            # Could start async LLM call here for even lower latency
            # For now, just mark that we predicted early
    
    def _on_turn_predicted_complete(self, text: str):
        """Called when turn is confirmed complete"""
        pass  # Main processing happens in _process_transcript
    
    async def _timeout_loop(self):
        """Monitor for session timeout"""
        while self._running:
            await asyncio.sleep(1.0)
            
            if self.verifier and self.verifier.check_timeout():
                print("[SESSION] Timeout - session ended")
                self.dialogue.reset_session()
                
                # Reset BT state
                if BT_AVAILABLE and self.behavior_tree:
                    bb = get_blackboard()
                    bb.session_state = SessionState.IDLE
                    bb.target_user_id = None
    
    def _on_vad_change(self, is_speech: bool, probability: float):
        """Callback for VAD state changes"""
        # Could feed to Semantic VAD here for real-time analysis
        pass
    
    def _on_transcript(self, transcript: TranscriptResult):
        """Callback for final transcripts"""
        # Feed partial text to turn manager for prediction
        if transcript.text:
            self.turn_manager.process_partial(transcript.text)
    
    def _on_response(self, response: DialogueResponse):
        """Callback for dialogue responses"""
        self._last_activity = time.time()


async def main():
    """Main entry point"""
    system = HRISystem()
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        print("\n[SIGNAL] Interrupt received")
        asyncio.create_task(system.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows
    
    try:
        await system.start()
        
        while system._running:
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        await system.stop()


def run():
    """Convenience function to run the system"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
