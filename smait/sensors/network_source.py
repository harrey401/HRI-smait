"""
SMAIT HRI v2.0 - Jackie Robot Network Bridge
Wraps NetworkAudioSource and NetworkVideoSource with WebSocket-based
bidirectional communication for the Jackie Android app.
"""

import asyncio
import threading
import json
import time
import struct
from typing import Optional, Tuple
import numpy as np

from smait.sensors.sources import AudioSource, VideoSource, set_jackie_audio_source, set_jackie_video_source
from smait.sensors.network_sources import NetworkAudioSource, NetworkVideoSource


class JackieAudioSource(AudioSource):
    """Audio source that receives from Jackie Android app via WebSocket"""
    
    def __init__(self, ws_server, sample_rate: int = 16000):
        self._ws_server = ws_server
        self._sample_rate = sample_rate
        self._active = False
        self._net_audio = NetworkAudioSource(sample_rate=sample_rate)
    
    def start(self):
        self._net_audio.start()
        self._active = True
    
    def stop(self):
        self._net_audio.stop()
        self._active = False
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        return self._net_audio.read(num_samples)
    
    def read_nonblocking(self) -> Optional[np.ndarray]:
        return self._net_audio.read_nonblocking()
    
    def push_audio(self, data: bytes):
        """Push raw audio bytes from WebSocket"""
        audio = np.frombuffer(data, dtype=np.int16)
        try:
            self._net_audio._buffer.put_nowait(audio)
        except:
            try:
                self._net_audio._buffer.get_nowait()
                self._net_audio._buffer.put_nowait(audio)
            except:
                pass
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def is_active(self) -> bool:
        return self._active


class JackieVideoSource(VideoSource):
    """Video source that receives from Jackie Android app via WebSocket"""
    
    def __init__(self, ws_server, width: int = 1280, height: int = 720):
        self._ws_server = ws_server
        self._width = width
        self._height = height
        self._active = False
        self._net_video = NetworkVideoSource(width=width, height=height)
    
    def start(self):
        self._active = True
        # Activate the internal buffer so read() returns frames
        self._net_video._active = True
    
    def stop(self):
        self._active = False
        self._net_video._active = False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._net_video.read()
    
    def push_frame(self, frame: np.ndarray):
        """Push frame from WebSocket"""
        self._net_video.push_frame(frame)
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)
    
    @property
    def is_active(self) -> bool:
        return self._active


class JackieWebSocketServer:
    """WebSocket server for Jackie Android app communication"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.audio_source: Optional[JackieAudioSource] = None
        self.video_source: Optional[JackieVideoSource] = None
        self._server = None
        self._loop = None
        self._thread = None
        self._clients = set()
    
    async def _handler(self, websocket):
        """Handle incoming WebSocket connections"""
        import websockets
        import base64
        self._clients.add(websocket)
        addr = websocket.remote_address
        print(f"[JACKIE] Client connected: {addr}")
        
        _msg_count = 0
        _audio_count = 0
        try:
            async for message in websocket:
                _msg_count += 1
                if _msg_count % 500 == 1:
                    print(f"[JACKIE] msg #{_msg_count}: type={'binary' if isinstance(message, bytes) else 'text'}, len={len(message)}, audio_src={self.audio_source is not None}")
                if isinstance(message, bytes):
                    # Binary message - efficient protocol (0x01=audio, 0x02=video)
                    if len(message) > 0:
                        msg_type = message[0]
                        payload = message[1:]
                        
                        if msg_type == 0x01 and self.audio_source:
                            _audio_count += 1
                            if _audio_count % 500 == 1:
                                print(f"[JACKIE] audio #{_audio_count}: {len(payload)} bytes")
                            self.audio_source.push_audio(payload)
                        elif msg_type == 0x02 and self.video_source:
                            import cv2
                            frame = cv2.imdecode(
                                np.frombuffer(payload, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            if frame is not None:
                                self.video_source.push_frame(frame)
                        elif msg_type == 0x03:
                            # DOA angle (4 bytes float)
                            if len(payload) >= 4:
                                import struct
                                angle = struct.unpack('>f', payload[:4])[0]
                                self._last_doa_angle = angle
                else:
                    # Text/JSON message - legacy protocol (Base64) + control commands
                    try:
                        cmd = json.loads(message)
                        msg_type = cmd.get("type", "")
                        
                        if msg_type == "audio" and self.audio_source:
                            # Legacy: JSON + Base64 audio
                            audio_b64 = cmd.get("data", "")
                            if audio_b64:
                                audio_bytes = base64.b64decode(audio_b64)
                                self.audio_source.push_audio(audio_bytes)
                        elif msg_type == "video" and self.video_source:
                            # Legacy: JSON + Base64 video
                            video_b64 = cmd.get("data", "")
                            if video_b64:
                                import cv2
                                jpeg_bytes = base64.b64decode(video_b64)
                                frame = cv2.imdecode(
                                    np.frombuffer(jpeg_bytes, dtype=np.uint8),
                                    cv2.IMREAD_COLOR
                                )
                                if frame is not None:
                                    self.video_source.push_frame(frame)
                        else:
                            # Control command
                            await self._handle_command(websocket, cmd)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[JACKIE] Client error: {e}")
        finally:
            self._clients.discard(websocket)
            print(f"[JACKIE] Client disconnected: {addr}")
    
    async def _handle_command(self, websocket, cmd: dict):
        """Handle control commands from Android app"""
        cmd_type = cmd.get("type", "")
        
        if cmd_type == "ping":
            await websocket.send(json.dumps({"type": "pong", "time": time.time()}))
        elif cmd_type == "get_config":
            await websocket.send(json.dumps({
                "type": "config",
                "audio_rate": 16000,
                "video_width": 1280,
                "video_height": 720,
                "use_cae": True
            }))
        elif cmd_type == "config":
            from smait.core.config import get_config
            cfg = get_config()
            if "vad_threshold" in cmd:
                cfg.audio.vad_threshold = float(cmd["vad_threshold"])
                # Live-update the VAD instance if audio pipeline is running
                from smait.sensors.audio_pipeline import get_audio_pipeline
                pipeline = get_audio_pipeline()
                if pipeline and pipeline.vad:
                    pipeline.vad.threshold = cfg.audio.vad_threshold
                print(f"[CONFIG] VAD threshold → {cfg.audio.vad_threshold:.2f}")
            if "asd_min_score" in cmd:
                cfg.vision.asd_min_score = float(cmd["asd_min_score"])
                print(f"[CONFIG] ASD min score → {cfg.vision.asd_min_score:.2f}")
            if "session_timeout" in cmd:
                cfg.session.timeout_seconds = int(cmd["session_timeout"])
                print(f"[CONFIG] Session timeout → {cfg.session.timeout_seconds}s")
            await websocket.send(json.dumps({"type": "config_ack", "status": "ok"}))
        elif cmd_type == "doa":
            # Direction of Arrival from CAE SDK
            angle = cmd.get("angle", 0)
            self._last_doa_angle = angle
            if hasattr(self, '_doa_callback') and self._doa_callback:
                self._doa_callback(angle)
        elif cmd_type == "cae_status":
            # CAE SDK status from Android
            print(f"[JACKIE] CAE status: AEC={cmd.get('aec', False)}, "
                  f"BF={cmd.get('beamforming', False)}, "
                  f"NS={cmd.get('noise_suppression', False)}")
    
    def _dispatch(self, coro):
        """Thread-safe: schedule a coroutine on the server's own event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ── public sync API (safe to call from any thread/loop) ──────────────────

    def send_tts(self, text: str):
        """Send TTS text to Jackie for playback via Android TTS"""
        self._dispatch(self._send_tts_async(text))

    def send_tts_audio(self, audio_data: bytes):
        """Send TTS audio bytes back to Jackie for playback"""
        self._dispatch(self._send_tts_audio_async(audio_data))

    def send_response(self, text: str):
        """Send text response to Jackie (for display on screen)"""
        self._dispatch(self._send_response_async(text))

    def send_transcript(self, text: str, is_user: bool = True):
        """Send transcript to Jackie (for display on screen)"""
        self._dispatch(self._send_transcript_async(text, is_user))

    def send_state(self, state: str):
        """Send session state to Jackie (for UI updates)"""
        self._dispatch(self._send_state_async(state))

    def send_photo_command(self):
        """Tell Jackie to take a selfie photo"""
        self._dispatch(self._send_photo_async())

    # ── private async implementations (run on server's own loop) ─────────────

    async def _send_tts_async(self, text: str):
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps({"type": "tts", "text": text}))
            except Exception:
                pass

    async def _send_tts_audio_async(self, audio_data: bytes):
        for ws in list(self._clients):
            try:
                await ws.send(b'\x04' + audio_data)
            except Exception:
                pass

    async def _send_response_async(self, text: str):
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps({"type": "response", "text": text}))
            except Exception:
                pass

    async def _send_transcript_async(self, text: str, is_user: bool = True):
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps({
                    "type": "transcript",
                    "text": text,
                    "speaker": "user" if is_user else "robot"
                }))
            except Exception:
                pass

    async def _send_state_async(self, state: str):
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps({"type": "state", "state": state}))
            except Exception:
                pass

    async def _send_photo_async(self):
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps({"type": "take_photo"}))
            except Exception:
                pass
    
    async def _run(self):
        """Run the WebSocket server"""
        import websockets
        self._server = await websockets.serve(self._handler, self.host, self.port)
        await self._server.wait_closed()
    
    def start(self):
        """Start server in background thread"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()
    
    def _run_in_thread(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())
    
    def stop(self):
        if self._server:
            self._server.close()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)


# ============================================================================
# Jackie initialization helpers (used by run_jackie.py)
# ============================================================================

_jackie_server: Optional[JackieWebSocketServer] = None
_jackie_audio: Optional[JackieAudioSource] = None
_jackie_video: Optional[JackieVideoSource] = None


def init_jackie_sources(host: str = "0.0.0.0", port: int = 8765):
    """Initialize Jackie WebSocket server and return audio/video sources"""
    global _jackie_server, _jackie_audio, _jackie_video
    
    _jackie_server = JackieWebSocketServer(host=host, port=port)
    _jackie_audio = JackieAudioSource(_jackie_server)
    _jackie_video = JackieVideoSource(_jackie_server)
    
    _jackie_server.audio_source = _jackie_audio
    _jackie_server.video_source = _jackie_video
    
    _jackie_server.start()
    _jackie_audio.start()
    
    print(f"[JACKIE] WebSocket server started on ws://{host}:{port}")
    
    return _jackie_audio, _jackie_video


def shutdown_jackie():
    """Shutdown Jackie server and sources"""
    global _jackie_server, _jackie_audio, _jackie_video
    
    if _jackie_audio:
        _jackie_audio.stop()
    if _jackie_video:
        _jackie_video.stop()
    if _jackie_server:
        _jackie_server.stop()
    
    print("[JACKIE] Shutdown complete")


def get_jackie_server() -> Optional[JackieWebSocketServer]:
    """Get the running Jackie server instance"""
    return _jackie_server
