"""
SMAIT HRI System v2.0 - Audio Processing Pipeline
Features:
- Non-blocking callback-based audio capture
- Voice Activity Detection (Silero VAD)
- Ring buffer with timestamp alignment
- Async speech segment generation
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Callable, List
import numpy as np

from smait.core.config import get_config
from smait.core.events import AudioChunk, SpeechSegment
from smait.sensors.sources import AudioSource, create_audio_source


@dataclass
class TimestampedAudio:
    """Audio chunk with precise timestamp"""
    samples: np.ndarray
    timestamp: float  # Wall-clock time at capture
    is_speech: bool = False
    speech_prob: float = 0.0


class AudioRingBuffer:
    """
    Thread-safe ring buffer for audio with timestamps.
    Allows querying audio by wall-clock time range.
    """
    
    def __init__(self, duration_seconds: float = 30.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(duration_seconds * sample_rate)
        
        # Circular buffer for samples
        self.buffer = np.zeros(self.max_samples, dtype=np.int16)
        self.write_pos = 0
        self.total_written = 0
        
        # Timestamp tracking (store timestamp at regular intervals)
        self.timestamp_interval = sample_rate // 10  # Every 100ms
        self.timestamps: deque = deque(maxlen=int(duration_seconds * 10))
        
        self.lock = threading.Lock()
    
    def write(self, samples: np.ndarray, timestamp: float):
        """Write samples to buffer with timestamp"""
        with self.lock:
            n = len(samples)
            
            # Record timestamp at write position
            self.timestamps.append({
                'pos': self.write_pos,
                'time': timestamp,
                'samples': n
            })
            
            if n >= self.max_samples:
                # Input larger than buffer - just keep the latest
                self.buffer[:] = samples[-self.max_samples:]
                self.write_pos = 0
                self.total_written = self.max_samples
                return
            
            # Write with wraparound
            space = self.max_samples - self.write_pos
            if n <= space:
                self.buffer[self.write_pos:self.write_pos + n] = samples
            else:
                self.buffer[self.write_pos:] = samples[:space]
                self.buffer[:n - space] = samples[space:]
            
            self.write_pos = (self.write_pos + n) % self.max_samples
            self.total_written += n
    
    def get_range(self, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Get audio samples between two wall-clock times"""
        with self.lock:
            if not self.timestamps:
                return None
            
            # Find sample positions for time range
            start_pos = None
            end_pos = None
            
            for ts in self.timestamps:
                if ts['time'] <= start_time:
                    start_pos = ts['pos']
                if ts['time'] <= end_time:
                    end_pos = (ts['pos'] + ts['samples']) % self.max_samples
            
            if start_pos is None or end_pos is None:
                return None
            
            # Extract samples (handle wraparound)
            if end_pos > start_pos:
                return self.buffer[start_pos:end_pos].copy()
            else:
                return np.concatenate([
                    self.buffer[start_pos:],
                    self.buffer[:end_pos]
                ]).copy()
    
    def get_last_seconds(self, seconds: float) -> np.ndarray:
        """Get the last N seconds of audio"""
        n_samples = min(int(seconds * self.sample_rate), self.total_written, self.max_samples)
        
        with self.lock:
            if n_samples == 0:
                return np.array([], dtype=np.int16)
            
            start = (self.write_pos - n_samples) % self.max_samples
            
            if start < self.write_pos:
                return self.buffer[start:self.write_pos].copy()
            return np.concatenate([
                self.buffer[start:],
                self.buffer[:self.write_pos]
            ]).copy()


class VoiceActivityDetector:
    """
    Voice Activity Detection using Silero VAD.
    Falls back to energy-based VAD if Silero unavailable.
    """
    
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self._use_silero = False
        
        self._load_model()
    
    def _load_model(self):
        """Try to load Silero VAD model"""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True  # Use ONNX for faster CPU inference
            )
            self.model = model
            self._get_speech_timestamps = utils[0]
            self._use_silero = True
            print("[VAD] Silero VAD loaded (ONNX)")
        except Exception as e:
            print(f"[VAD] Silero unavailable ({e}), using energy-based VAD")
            self._use_silero = False
    
    def process_chunk(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Process audio chunk and return (is_speech, probability).
        Audio should be int16 or float32, 16kHz.
        """
        if len(audio) == 0:
            return False, 0.0
        
        if self._use_silero:
            return self._silero_vad(audio)
        else:
            return self._energy_vad(audio)
    
    def _silero_vad(self, audio: np.ndarray) -> tuple[bool, float]:
        """Silero VAD inference"""
        import torch
        
        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        # Silero VAD is VERY strict: exactly 512 samples for 16kHz
        # Process in 512-sample chunks and average the probabilities
        chunk_size = 512
        
        if len(audio_float) < chunk_size:
            # Pad short audio
            audio_float = np.pad(audio_float, (0, chunk_size - len(audio_float)))
        
        # Process chunks and get max probability
        probs = []
        for i in range(0, len(audio_float) - chunk_size + 1, chunk_size):
            chunk = audio_float[i:i + chunk_size]
            tensor = torch.from_numpy(chunk)
            
            with torch.no_grad():
                prob = self.model(tensor, self.sample_rate).item()
                probs.append(prob)
        
        # If we have leftover samples that don't make a full chunk, skip them
        # (they'll be processed in the next call)
        
        if not probs:
            return False, 0.0
        
        # Use max probability (if any chunk has speech, consider it speech)
        max_prob = max(probs)
        return max_prob > self.threshold, max_prob
    
    def _energy_vad(self, audio: np.ndarray) -> tuple[bool, float]:
        """Simple energy-based VAD fallback"""
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio
        
        # RMS energy
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Normalize to 0-1 range (rough approximation)
        prob = min(1.0, rms * 20)
        
        # Dynamic threshold based on recent history
        # (simplified - real implementation would track noise floor)
        return prob > 0.02, prob


class AudioPipeline:
    """
        Main audio processing pipeline.
    Captures audio, runs VAD, and yields speech segments.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Components
        self.source: Optional[AudioSource] = None
        self.vad = VoiceActivityDetector(
            threshold=self.config.audio.vad_threshold,
            sample_rate=self.config.audio.sample_rate
        )
        self.ring_buffer = AudioRingBuffer(
            duration_seconds=self.config.audio.buffer_seconds,
            sample_rate=self.config.audio.sample_rate
        )
        
        # State
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        # Speech segment detection
        self._in_speech = False
        self._speech_start_time: Optional[float] = None
        self._speech_chunks: List[np.ndarray] = []
        self._silence_samples = 0
        
        # Output queue for async iteration
        self._segment_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        
        # Callbacks
        self._on_vad_change: Optional[Callable[[bool, float], None]] = None
        self._on_speech_start: Optional[Callable[[], None]] = None
        self._on_speech_end: Optional[Callable[[], None]] = None
    
    def set_vad_callback(self, callback: Callable[[bool, float], None]):
        """Set callback for VAD state changes (for UI feedback)"""
        self._on_vad_change = callback
    
    def set_speech_start_callback(self, callback: Callable[[], None]):
        """Set callback for when speech starts (for ASD sync)"""
        self._on_speech_start = callback
    
    def set_speech_end_callback(self, callback: Callable[[], None]):
        """Set callback for when speech ends (for ASD sync)"""
        self._on_speech_end = callback
    
    def start(self):
        """Start audio capture and processing"""
        self.source = create_audio_source()
        self.source.start()
        
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="AudioCapture"
        )
        self._capture_thread.start()
        print("[AUDIO] Pipeline started")
    
    def stop(self):
        """Stop audio capture"""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self.source:
            self.source.stop()
        print("[AUDIO] Pipeline stopped")
    
    def _capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        chunk_samples = int(
            self.config.audio.sample_rate * 
            self.config.audio.chunk_duration_ms / 1000
        )
        silence_threshold_samples = int(
            self.config.audio.sample_rate *
            self.config.audio.silence_duration_ms / 1000
        )
        min_speech_samples = int(
            self.config.audio.sample_rate *
            self.config.audio.min_speech_duration_ms / 1000
        )
        
        while self._running:
            # Read audio chunk
            audio = self.source.read_nonblocking()
            if audio is None or len(audio) == 0:
                time.sleep(0.01)
                continue
            
            timestamp = time.time()
            
            # Write to ring buffer
            self.ring_buffer.write(audio, timestamp)
            
            # Run VAD
            is_speech, prob = self.vad.process_chunk(audio)
            
            # Callback for UI
            if self._on_vad_change:
                self._on_vad_change(is_speech, prob)
            
            # Speech segment detection state machine
            if is_speech:
                if not self._in_speech:
                    # Speech start
                    self._in_speech = True
                    self._speech_start_time = timestamp
                    self._speech_chunks = []
                    self._silence_samples = 0
                    
                    # Callback for ASD synchronization
                    if self._on_speech_start:
                        self._on_speech_start()
                
                self._speech_chunks.append(audio)
                self._silence_samples = 0
                
            else:
                if self._in_speech:
                    self._silence_samples += len(audio)
                    self._speech_chunks.append(audio)  # Include trailing silence
                    
                    # Check if silence duration exceeded
                    if self._silence_samples >= silence_threshold_samples:
                        # End of speech segment
                        speech_audio = np.concatenate(self._speech_chunks)
                        
                        # Callback for ASD synchronization
                        if self._on_speech_end:
                            self._on_speech_end()
                        
                        # Only emit if long enough
                        if len(speech_audio) >= min_speech_samples:
                            segment = SpeechSegment(
                                audio=speech_audio,
                                start_time=self._speech_start_time,
                                end_time=timestamp
                            )
                            
                            # Put in async queue
                            try:
                                self._segment_queue.put_nowait(segment)
                            except asyncio.QueueFull:
                                pass  # Drop if queue full
                        
                        # Reset state
                        self._in_speech = False
                        self._speech_start_time = None
                        self._speech_chunks = []
    
    async def speech_segments(self) -> AsyncIterator[SpeechSegment]:
        """Async iterator yielding speech segments"""
        while self._running:
            try:
                segment = await asyncio.wait_for(
                    self._segment_queue.get(),
                    timeout=0.5
                )
                yield segment
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    def get_audio_for_time(self, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Get audio samples for a specific time range"""
        return self.ring_buffer.get_range(start_time, end_time)
    
    def get_recent_audio(self, seconds: float = 2.0) -> np.ndarray:
        """Get the most recent N seconds of audio"""
        return self.ring_buffer.get_last_seconds(seconds)
