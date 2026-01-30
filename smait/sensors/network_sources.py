"""
SMAIT HRI System v2.0 - Network Streaming Sources
Receives video/audio streams from Android robot app over network.
"""

import threading
import queue
import time
import socket
import struct
from typing import Optional, Tuple, Callable
import numpy as np
import cv2

from smait.sensors.sources import VideoSource, AudioSource
from smait.core.config import get_config


class NetworkVideoSource(VideoSource):
    """
    Receives video frames from Android app over TCP/UDP.

    Protocol options:
    - MJPEG over HTTP (simple, Android-friendly)
    - Raw frames over TCP (lower latency)
    - WebSocket (bidirectional)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        width: int = 1280,
        height: int = 720,
        protocol: str = "mjpeg"  # mjpeg, raw, websocket
    ):
        self._host = host
        self._port = port
        self._width = width
        self._height = height
        self._protocol = protocol

        self._active = False
        self._frame_buffer: queue.Queue = queue.Queue(maxsize=5)
        self._latest_frame: Optional[np.ndarray] = None
        self._receive_thread: Optional[threading.Thread] = None

        # Stats
        self._frames_received = 0
        self._last_frame_time = 0

    def start(self):
        """Start receiving video stream"""
        self._active = True

        if self._protocol == "mjpeg":
            self._receive_thread = threading.Thread(
                target=self._receive_mjpeg,
                daemon=True,
                name="NetworkVideo"
            )
        else:
            self._receive_thread = threading.Thread(
                target=self._receive_raw,
                daemon=True,
                name="NetworkVideo"
            )

        self._receive_thread.start()
        print(f"[NET-VIDEO] Listening on {self._host}:{self._port} ({self._protocol})")

    def stop(self):
        """Stop receiving video stream"""
        self._active = False
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        print("[NET-VIDEO] Stopped")

    def _receive_mjpeg(self):
        """Receive MJPEG stream (HTTP multipart)"""
        import urllib.request

        # Wait for Android app to connect and stream
        # This assumes Android is pushing to this URL
        # Alternative: run HTTP server and accept connections

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(1.0)

        print(f"[NET-VIDEO] Waiting for connection on port {self._port}...")

        while self._active:
            try:
                conn, addr = server.accept()
                print(f"[NET-VIDEO] Connected from {addr}")
                self._handle_mjpeg_connection(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self._active:
                    print(f"[NET-VIDEO] Error: {e}")
                time.sleep(0.1)

        server.close()

    def _handle_mjpeg_connection(self, conn: socket.socket):
        """Handle MJPEG stream from connected client"""
        buffer = b""

        while self._active:
            try:
                data = conn.recv(65536)
                if not data:
                    break

                buffer += data

                # Find JPEG boundaries (FFD8 = start, FFD9 = end)
                start = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9')

                if start != -1 and end != -1 and end > start:
                    jpg_data = buffer[start:end+2]
                    buffer = buffer[end+2:]

                    # Decode JPEG
                    frame = cv2.imdecode(
                        np.frombuffer(jpg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is not None:
                        self._push_frame(frame)

            except Exception as e:
                if self._active:
                    print(f"[NET-VIDEO] Stream error: {e}")
                break

        conn.close()

    def _receive_raw(self):
        """Receive raw frames over TCP"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(1.0)

        while self._active:
            try:
                conn, addr = server.accept()
                print(f"[NET-VIDEO] Raw connection from {addr}")
                self._handle_raw_connection(conn)
            except socket.timeout:
                continue
            except Exception as e:
                if self._active:
                    print(f"[NET-VIDEO] Error: {e}")

        server.close()

    def _handle_raw_connection(self, conn: socket.socket):
        """Handle raw frame stream (header + data)"""
        while self._active:
            try:
                # Read header: width (4), height (4), size (4)
                header = self._recv_exact(conn, 12)
                if not header:
                    break

                width, height, size = struct.unpack('>III', header)

                # Read frame data
                data = self._recv_exact(conn, size)
                if not data:
                    break

                # Decode (assuming JPEG)
                frame = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    self._push_frame(frame)

            except Exception as e:
                if self._active:
                    print(f"[NET-VIDEO] Error: {e}")
                break

        conn.close()

    def _recv_exact(self, conn: socket.socket, size: int) -> Optional[bytes]:
        """Receive exactly size bytes"""
        data = b""
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _push_frame(self, frame: np.ndarray):
        """Push frame to buffer"""
        self._frames_received += 1
        self._last_frame_time = time.time()
        self._latest_frame = frame

        try:
            self._frame_buffer.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_buffer.get_nowait()
                self._frame_buffer.put_nowait(frame)
            except:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame"""
        if not self._active:
            return False, None

        try:
            frame = self._frame_buffer.get(timeout=0.1)
            return True, frame
        except queue.Empty:
            if self._latest_frame is not None:
                return True, self._latest_frame
            return False, None

    def push_frame(self, frame: np.ndarray):
        """External push (for WebSocket or other protocols)"""
        self._push_frame(frame)

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def fps(self) -> float:
        """Estimated FPS based on received frames"""
        if self._frames_received < 2:
            return 0.0
        # Simple estimate
        return self._frames_received / max(1, time.time() - self._last_frame_time + 0.001)


class NetworkAudioSource(AudioSource):
    """
    Receives audio stream from Android app over network.

    Expected format: 16kHz, mono, int16
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5001,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        self._host = host
        self._port = port
        self._sample_rate = sample_rate
        self._channels = channels

        self._active = False
        self._buffer: queue.Queue = queue.Queue(maxsize=1000)
        self._receive_thread: Optional[threading.Thread] = None

    def start(self):
        """Start receiving audio stream"""
        self._active = True
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name="NetworkAudio"
        )
        self._receive_thread.start()
        print(f"[NET-AUDIO] Listening on {self._host}:{self._port}")

    def stop(self):
        """Stop receiving audio stream"""
        self._active = False
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        print("[NET-AUDIO] Stopped")

    def _receive_loop(self):
        """Receive audio data over UDP (low latency)"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.settimeout(0.1)

        print(f"[NET-AUDIO] Waiting for audio on port {self._port}...")

        while self._active:
            try:
                data, addr = sock.recvfrom(65536)

                # Convert to int16 array
                audio = np.frombuffer(data, dtype=np.int16)

                try:
                    self._buffer.put_nowait(audio)
                except queue.Full:
                    try:
                        self._buffer.get_nowait()
                        self._buffer.put_nowait(audio)
                    except:
                        pass

            except socket.timeout:
                continue
            except Exception as e:
                if self._active:
                    print(f"[NET-AUDIO] Error: {e}")

        sock.close()

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read specified number of samples"""
        if not self._active:
            return None

        collected = []
        total = 0

        while total < num_samples:
            try:
                chunk = self._buffer.get(timeout=0.5)
                collected.append(chunk)
                total += len(chunk)
            except queue.Empty:
                break

        if not collected:
            return None

        audio = np.concatenate(collected)
        return audio[:num_samples] if len(audio) >= num_samples else audio

    def read_nonblocking(self) -> Optional[np.ndarray]:
        """Read all available samples"""
        if not self._active:
            return None

        collected = []
        while True:
            try:
                chunk = self._buffer.get_nowait()
                collected.append(chunk)
            except queue.Empty:
                break

        if not collected:
            return None

        return np.concatenate(collected)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_active(self) -> bool:
        return self._active
