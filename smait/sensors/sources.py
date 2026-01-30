"""
SMAIT HRI System v2.0 - Sensor Abstraction Layer
Provides unified interface for cameras and microphones across:
- Real hardware (laptop/robot)
- Isaac Sim simulation
- ROS 2 topics
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, AsyncIterator, Tuple
import asyncio
import threading
import queue
import time
import numpy as np

from smait.core.config import get_config, DeploymentMode
from smait.core.events import AudioChunk, FrameResult


# ============================================================================
# Abstract Base Classes
# ============================================================================

class AudioSource(ABC):
    """Abstract audio input source"""
    
    @abstractmethod
    def start(self):
        """Start capturing audio"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop capturing audio"""
        pass
    
    @abstractmethod
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read audio samples (blocking)"""
        pass
    
    @abstractmethod
    def read_nonblocking(self) -> Optional[np.ndarray]:
        """Read available audio samples (non-blocking)"""
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass
    
    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass


class VideoSource(ABC):
    """Abstract video input source"""
    
    @abstractmethod
    def start(self):
        """Start capturing video"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop capturing video"""
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame (blocking)"""
        pass
    
    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height)"""
        pass
    
    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass


# ============================================================================
# Real Hardware Implementations
# ============================================================================

class MicrophoneSource(AudioSource):
    """Real microphone using sounddevice with callback-based capture"""
    
    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_seconds: float = 30.0
    ):
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._buffer_size = int(sample_rate * buffer_seconds)
        
        self._buffer: queue.Queue = queue.Queue(maxsize=1000)
        self._stream = None
        self._active = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice - runs in separate thread"""
        if status:
            print(f"[MIC] Status: {status}")
        
        # Convert to int16 and queue
        audio = (np.clip(indata[:, 0], -1.0, 1.0) * 32767).astype(np.int16)
        try:
            self._buffer.put_nowait(audio)
        except queue.Full:
            # Drop oldest if buffer full
            try:
                self._buffer.get_nowait()
                self._buffer.put_nowait(audio)
            except:
                pass
    
    def start(self):
        """Start microphone capture with callback"""
        import sounddevice as sd
        
        self._stream = sd.InputStream(
            device=self._device,
            channels=self._channels,
            samplerate=self._sample_rate,
            dtype='float32',
            blocksize=int(self._sample_rate * 0.03),  # 30ms chunks
            callback=self._audio_callback
        )
        self._stream.start()
        self._active = True
        print(f"[MIC] Started (device={self._device}, rate={self._sample_rate})")
    
    def stop(self):
        """Stop microphone capture"""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False
        print("[MIC] Stopped")
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read specified number of samples (blocking, accumulates chunks)"""
        if not self._active:
            return None
        
        collected = []
        total = 0
        
        while total < num_samples:
            try:
                chunk = self._buffer.get(timeout=1.0)
                collected.append(chunk)
                total += len(chunk)
            except queue.Empty:
                break
        
        if not collected:
            return None
        
        audio = np.concatenate(collected)
        return audio[:num_samples] if len(audio) >= num_samples else audio
    
    def read_nonblocking(self) -> Optional[np.ndarray]:
        """Read all available samples without blocking"""
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


class CameraSource(VideoSource):
    """Real camera using OpenCV"""
    
    def __init__(
        self,
        device: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None
        self._active = False
    
    def start(self):
        """Start camera capture"""
        import cv2
        
        # Try different backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for backend, name in backends:
            try:
                self._cap = cv2.VideoCapture(self._device, backend)
                if self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        print(f"[CAMERA] Using {name} backend")
                        break
                self._cap.release()
            except:
                pass
        
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self._device}")
        
        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        
        # Get actual resolution
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = actual_w
        self._height = actual_h
        
        self._active = True
        print(f"[CAMERA] Started ({actual_w}x{actual_h})")
    
    def stop(self):
        """Stop camera capture"""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._active = False
        print("[CAMERA] Stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame"""
        if not self._cap or not self._active:
            return False, None
        return self._cap.read()
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)
    
    @property
    def is_active(self) -> bool:
        return self._active


# ============================================================================
# Isaac Sim Implementations
# ============================================================================

class IsaacSimAudioSource(AudioSource):
    """
    Audio source for Isaac Sim.
    Since Isaac Sim doesn't simulate audio, this uses:
    1. Pre-recorded audio files for testing
    2. Synthetic speech generation
    3. Real microphone input synced with sim time
    """
    
    def __init__(
        self,
        audio_file: Optional[str] = None,
        sample_rate: int = 16000,
        use_real_mic: bool = False
    ):
        self._sample_rate = sample_rate
        self._audio_file = audio_file
        self._use_real_mic = use_real_mic
        
        self._audio_data: Optional[np.ndarray] = None
        self._position = 0
        self._active = False
        
        # Fallback to real mic
        self._real_mic: Optional[MicrophoneSource] = None
        
        if audio_file:
            self._load_audio_file(audio_file)
    
    def _load_audio_file(self, path: str):
        """Load audio file for playback"""
        try:
            import soundfile as sf
            self._audio_data, sr = sf.read(path, dtype='int16')
            if sr != self._sample_rate:
                # Resample if needed
                import scipy.signal as signal
                num_samples = int(len(self._audio_data) * self._sample_rate / sr)
                self._audio_data = signal.resample(self._audio_data, num_samples).astype(np.int16)
            print(f"[ISAAC_AUDIO] Loaded {path} ({len(self._audio_data)/self._sample_rate:.1f}s)")
        except Exception as e:
            print(f"[ISAAC_AUDIO] Failed to load {path}: {e}")
    
    def start(self):
        if self._use_real_mic:
            self._real_mic = MicrophoneSource(sample_rate=self._sample_rate)
            self._real_mic.start()
        self._position = 0
        self._active = True
        print("[ISAAC_AUDIO] Started")
    
    def stop(self):
        if self._real_mic:
            self._real_mic.stop()
        self._active = False
        print("[ISAAC_AUDIO] Stopped")
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        if not self._active:
            return None
        
        if self._use_real_mic and self._real_mic:
            return self._real_mic.read(num_samples)
        
        if self._audio_data is None:
            # Return silence
            return np.zeros(num_samples, dtype=np.int16)
        
        # Return chunk from loaded audio (loop if needed)
        end = self._position + num_samples
        if end <= len(self._audio_data):
            chunk = self._audio_data[self._position:end]
            self._position = end
        else:
            # Loop back
            chunk = np.concatenate([
                self._audio_data[self._position:],
                self._audio_data[:end - len(self._audio_data)]
            ])
            self._position = end - len(self._audio_data)
        
        return chunk
    
    def read_nonblocking(self) -> Optional[np.ndarray]:
        if self._use_real_mic and self._real_mic:
            return self._real_mic.read_nonblocking()
        # For file playback, return a standard chunk
        return self.read(int(self._sample_rate * 0.03))
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def is_active(self) -> bool:
        return self._active


class IsaacSimVideoSource(VideoSource):
    """
    Video source from Isaac Sim camera.
    Connects via ROS 2 or direct Omniverse API.
    """
    
    def __init__(
        self,
        camera_prim: str = "/World/Robot/Camera",
        width: int = 1280,
        height: int = 720
    ):
        self._camera_prim = camera_prim
        self._width = width
        self._height = height
        self._active = False
        
        # Frame buffer (populated by ROS callback or sim step)
        self._frame_buffer: queue.Queue = queue.Queue(maxsize=5)
        self._latest_frame: Optional[np.ndarray] = None
    
    def start(self):
        """
        Start receiving frames from Isaac Sim.
        In practice, this would:
        1. Subscribe to ROS 2 camera topic, or
        2. Connect to Omniverse streaming
        """
        self._active = True
        print(f"[ISAAC_VIDEO] Started (prim={self._camera_prim})")
        # Note: Actual Isaac Sim integration would happen in ROS 2 node
    
    def stop(self):
        self._active = False
        print("[ISAAC_VIDEO] Stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from buffer"""
        if not self._active:
            return False, None
        
        try:
            frame = self._frame_buffer.get(timeout=0.1)
            self._latest_frame = frame
            return True, frame
        except queue.Empty:
            # Return last frame if available
            if self._latest_frame is not None:
                return True, self._latest_frame
            return False, None
    
    def push_frame(self, frame: np.ndarray):
        """Push frame from external source (ROS callback)"""
        try:
            self._frame_buffer.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_buffer.get_nowait()
                self._frame_buffer.put_nowait(frame)
            except:
                pass
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)
    
    @property
    def is_active(self) -> bool:
        return self._active


# ============================================================================
# Factory Functions
# ============================================================================

def create_audio_source() -> AudioSource:
    """Create appropriate audio source based on config"""
    config = get_config()
    
    if config.mode == DeploymentMode.ISAAC_SIM:
        return IsaacSimAudioSource(
            audio_file=config.isaac_sim.synthetic_audio_path if config.isaac_sim.use_synthetic_audio else None,
            sample_rate=config.audio.sample_rate,
            use_real_mic=not config.isaac_sim.use_synthetic_audio
        )
    else:
        return MicrophoneSource(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels
        )


def create_video_source() -> VideoSource:
    """Create appropriate video source based on config"""
    config = get_config()
    
    if config.mode == DeploymentMode.ISAAC_SIM:
        return IsaacSimVideoSource(
            camera_prim=config.isaac_sim.camera_prim,
            width=config.vision.frame_width,
            height=config.vision.frame_height
        )
    else:
        return CameraSource(
            device=config.vision.camera_index,
            width=config.vision.frame_width,
            height=config.vision.frame_height,
            fps=config.vision.target_fps
        )
