"""
SMAIT HRI v2.0 - Audio-Visual Delay Injection

Introduces controlled temporal misalignment between audio and video streams
to test the system's tolerance to AV desync.

In real-world scenarios, AV desync can occur due to:
- Network latency differences
- Processing pipeline delays
- Camera/microphone hardware differences
- Buffering in different stages

Usage:
    injector = AVDelayInjector(delay_ms=100)  # Audio 100ms ahead of video
    
    # In video processing loop:
    delayed_frame = injector.delay_video(frame, timestamp)
    
    # In audio processing loop:
    delayed_audio = injector.delay_audio(audio_chunk, timestamp)
"""

import time
import threading
from collections import deque
from typing import Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TimestampedFrame:
    """Video frame with timestamp"""
    frame: np.ndarray
    timestamp: float
    frame_id: int


@dataclass
class TimestampedAudio:
    """Audio chunk with timestamp"""
    audio: np.ndarray
    timestamp: float
    sample_rate: int


class AVDelayInjector:
    """
    Injects controlled audio-visual delay for testing.
    
    Positive delay: Audio arrives before video (audio leads)
    Negative delay: Video arrives before audio (video leads)
    
    The system should handle delays up to ~200ms without significant
    accuracy degradation. Beyond that, the temporal buffering in the
    verifier should compensate.
    """
    
    def __init__(
        self,
        delay_ms: float = 0.0,
        buffer_duration_s: float = 1.0
    ):
        """
        Args:
            delay_ms: Delay to inject (positive = audio ahead of video)
            buffer_duration_s: How long to buffer frames/audio
        """
        self.delay_ms = delay_ms
        self.buffer_duration_s = buffer_duration_s
        
        # Separate buffers for video and audio
        self._video_buffer: deque = deque()
        self._audio_buffer: deque = deque()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metrics
        self._frames_processed = 0
        self._audio_chunks_processed = 0
    
    @property
    def delay_seconds(self) -> float:
        return self.delay_ms / 1000.0
    
    def set_delay(self, delay_ms: float):
        """Update delay value"""
        self.delay_ms = delay_ms
        print(f"[AV-DELAY] Set to {delay_ms:+.0f}ms")
    
    def delay_video(
        self, 
        frame: np.ndarray, 
        timestamp: Optional[float] = None,
        frame_id: int = 0
    ) -> Optional[np.ndarray]:
        """
        Buffer video frame and return appropriately delayed frame.
        
        If delay > 0 (audio ahead), video is delayed by that amount.
        If delay < 0 (video ahead), we don't delay video.
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Add frame to buffer
            self._video_buffer.append(TimestampedFrame(
                frame=frame.copy(),
                timestamp=timestamp,
                frame_id=frame_id
            ))
            
            # Calculate target release time
            if self.delay_ms > 0:
                # Video needs to be delayed
                target_time = timestamp + self.delay_seconds
            else:
                # Video releases immediately (or we could delay audio instead)
                target_time = timestamp
            
            # Find frame to release
            current_time = time.time()
            result_frame = None
            
            # Release frames that are past their target time
            while self._video_buffer:
                oldest = self._video_buffer[0]
                
                if self.delay_ms > 0:
                    release_time = oldest.timestamp + self.delay_seconds
                else:
                    release_time = oldest.timestamp
                
                if current_time >= release_time:
                    result_frame = self._video_buffer.popleft().frame
                    self._frames_processed += 1
                else:
                    break
            
            # Cleanup old frames
            self._cleanup_buffer(self._video_buffer, current_time)
            
            return result_frame
    
    def delay_audio(
        self,
        audio: np.ndarray,
        timestamp: Optional[float] = None,
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Buffer audio chunk and return appropriately delayed audio.
        
        If delay < 0 (video ahead), audio is delayed by |delay|.
        If delay > 0 (audio ahead), we don't delay audio.
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Add audio to buffer
            self._audio_buffer.append(TimestampedAudio(
                audio=audio.copy(),
                timestamp=timestamp,
                sample_rate=sample_rate
            ))
            
            # Calculate target release time
            if self.delay_ms < 0:
                # Audio needs to be delayed
                target_time = timestamp + abs(self.delay_seconds)
            else:
                # Audio releases immediately
                target_time = timestamp
            
            # Find audio to release
            current_time = time.time()
            result_audio = None
            
            while self._audio_buffer:
                oldest = self._audio_buffer[0]
                
                if self.delay_ms < 0:
                    release_time = oldest.timestamp + abs(self.delay_seconds)
                else:
                    release_time = oldest.timestamp
                
                if current_time >= release_time:
                    result_audio = self._audio_buffer.popleft().audio
                    self._audio_chunks_processed += 1
                else:
                    break
            
            # Cleanup old audio
            self._cleanup_buffer(self._audio_buffer, current_time)
            
            return result_audio
    
    def _cleanup_buffer(self, buffer: deque, current_time: float):
        """Remove items older than buffer duration"""
        cutoff = current_time - self.buffer_duration_s
        
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            'delay_ms': self.delay_ms,
            'video_buffer_size': len(self._video_buffer),
            'audio_buffer_size': len(self._audio_buffer),
            'frames_processed': self._frames_processed,
            'audio_chunks_processed': self._audio_chunks_processed
        }
    
    def reset(self):
        """Clear buffers and reset stats"""
        with self._lock:
            self._video_buffer.clear()
            self._audio_buffer.clear()
            self._frames_processed = 0
            self._audio_chunks_processed = 0


class SynchronizedAVStream:
    """
    Manages synchronized audio-video streaming with delay injection.
    
    This wraps both video and audio sources, applying consistent
    delay to simulate real-world AV desync scenarios.
    """
    
    def __init__(self, delay_ms: float = 0.0):
        self.injector = AVDelayInjector(delay_ms=delay_ms)
        
        self._video_frames: deque = deque(maxlen=300)  # ~10s at 30fps
        self._audio_chunks: deque = deque(maxlen=100)
        
        self._running = False
        self._lock = threading.Lock()
    
    def add_video_frame(self, frame: np.ndarray, timestamp: float):
        """Add a video frame to the stream"""
        with self._lock:
            self._video_frames.append((frame, timestamp))
    
    def add_audio_chunk(self, audio: np.ndarray, timestamp: float):
        """Add an audio chunk to the stream"""
        with self._lock:
            self._audio_chunks.append((audio, timestamp))
    
    def get_synchronized_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get a synchronized video frame and audio chunk.
        
        Returns (video_frame, audio_chunk) with appropriate delay applied.
        """
        video_frame = None
        audio_chunk = None
        
        with self._lock:
            if self._video_frames:
                frame, ts = self._video_frames.popleft()
                video_frame = self.injector.delay_video(frame, ts)
            
            if self._audio_chunks:
                audio, ts = self._audio_chunks.popleft()
                audio_chunk = self.injector.delay_audio(audio, ts)
        
        return video_frame, audio_chunk
    
    def set_delay(self, delay_ms: float):
        """Update delay value"""
        self.injector.set_delay(delay_ms)


def test_delay_injector():
    """Test the delay injector"""
    import cv2
    
    print("Testing AV Delay Injector")
    print("=" * 40)
    
    # Test positive delay (audio ahead)
    injector = AVDelayInjector(delay_ms=100)
    
    # Simulate video frames
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        
        timestamp = time.time()
        delayed = injector.delay_video(frame, timestamp, i)
        
        if delayed is not None:
            print(f"Released frame (original timestamp: {timestamp:.3f})")
        
        time.sleep(0.05)
    
    # Wait for remaining frames
    time.sleep(0.2)
    
    stats = injector.get_stats()
    print(f"\nStats: {stats}")
    
    print("\nDelay injector test complete!")


if __name__ == "__main__":
    test_delay_injector()
