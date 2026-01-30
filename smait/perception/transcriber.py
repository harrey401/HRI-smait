"""
SMAIT HRI System v2.0 - Speech Recognition (ASR)
Supports:
- Parakeet TDT (NeMo, streaming, RECOMMENDED)
- faster-whisper (local, fallback)
- AWS Transcribe (cloud, legacy)
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator, List, Callable
import numpy as np

from smait.core.config import get_config, ASRBackend
from smait.core.events import SpeechSegment, TranscriptResult
from smait.sensors.audio_pipeline import AudioPipeline


class ASREngine(ABC):
    """Abstract ASR engine interface"""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe audio segment synchronously"""
        pass
    
    @abstractmethod
    async def transcribe_async(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe audio segment asynchronously"""
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this engine supports real-time streaming"""
        pass


class ParakeetEngine(ASREngine):
    """
    Parakeet TDT ASR engine wrapper.
    Uses NVIDIA NeMo for state-of-the-art streaming ASR.
    
    This is the RECOMMENDED engine for lowest latency.
    """
    
    def __init__(self):
        self.config = get_config()
        self._engine = None
        self._loaded = False
        
        self._load_engine()
    
    def _load_engine(self):
        """Load the Parakeet engine"""
        try:
            from smait.perception.parakeet_asr import ParakeetTDTEngine
            
            self._engine = ParakeetTDTEngine(
                model_name=self.config.asr.parakeet_model,
                device=self.config.asr.device,
                streaming=self.config.asr.parakeet_streaming,
                chunk_seconds=self.config.asr.parakeet_chunk_seconds
            )
            self._loaded = self._engine.is_loaded
            
        except ImportError as e:
            print(f"[ASR] Failed to load Parakeet engine: {e}")
            print(f"[ASR] Install NeMo: pip install nemo_toolkit[asr]")
            self._loaded = False
        except Exception as e:
            print(f"[ASR] Parakeet initialization error: {e}")
            self._loaded = False
    
    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        if not self._loaded:
            return TranscriptResult(
                text="[Parakeet not loaded]",
                is_final=True,
                confidence=0.0,
                timestamp=time.time()
            )
        return self._engine.transcribe(audio)
    
    async def transcribe_async(self, audio: np.ndarray) -> TranscriptResult:
        if not self._loaded:
            return TranscriptResult(
                text="[Parakeet not loaded]",
                is_final=True,
                confidence=0.0,
                timestamp=time.time()
            )
        return await self._engine.transcribe_async(audio)
    
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


class FasterWhisperEngine(ASREngine):
    """
    faster-whisper ASR engine.
    Uses CTranslate2 for efficient CPU/GPU inference.
    """
    
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        beam_size: int = 5
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        
        self.model = None
        self._lock = threading.Lock()
        
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            # Determine device
            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except:
                    device = "cpu"
            
            # Determine compute type
            compute_type = self.compute_type
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"
            
            print(f"[ASR] Loading faster-whisper ({self.model_size}) on {device}...")
            
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type
            )
            
            print(f"[ASR] Model loaded successfully")
            
        except ImportError:
            print("[ASR] faster-whisper not installed!")
            print("[ASR] Install with: pip install faster-whisper")
            raise
        except Exception as e:
            print(f"[ASR] Failed to load model: {e}")
            raise
    
    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """
        Transcribe audio segment.
        Audio should be int16 or float32 at 16kHz.
        """
        start_time = time.time()
        
        if self.model is None:
            return TranscriptResult(
                text="[ASR model not loaded]",
                is_final=True,
                confidence=0.0,
                timestamp=start_time
            )
        
        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        with self._lock:
            try:
                segments, info = self.model.transcribe(
                    audio_float,
                    language=self.language if not self.language.endswith('.en') else None,
                    beam_size=self.beam_size,
                    vad_filter=True,  # Additional VAD filtering
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=200
                    )
                )
                
                # Collect all segments
                text_parts = []
                words = []
                total_confidence = 0.0
                segment_count = 0
                
                for segment in segments:
                    text_parts.append(segment.text.strip())
                    total_confidence += segment.avg_logprob
                    segment_count += 1
                    
                    # Word-level timestamps if available
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            words.append({
                                'word': word.word,
                                'start': word.start,
                                'end': word.end,
                                'probability': word.probability
                            })
                
                text = " ".join(text_parts).strip()
                
                # Convert log prob to confidence (rough approximation)
                avg_logprob = total_confidence / max(segment_count, 1)
                confidence = min(1.0, max(0.0, 1.0 + avg_logprob / 5.0))
                
                latency = (time.time() - start_time) * 1000
                
                return TranscriptResult(
                    text=text,
                    is_final=True,
                    confidence=confidence,
                    timestamp=time.time(),
                    words=words
                )
                
            except Exception as e:
                print(f"[ASR] Transcription error: {e}")
                return TranscriptResult(
                    text="",
                    is_final=True,
                    confidence=0.0,
                    timestamp=time.time()
                )
    
    async def transcribe_async(self, audio: np.ndarray) -> TranscriptResult:
        """Async wrapper - runs transcription in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio)
    
    def supports_streaming(self) -> bool:
        """faster-whisper doesn't support true streaming"""
        return False


class AWSTranscribeEngine(ASREngine):
    """
    AWS Transcribe streaming ASR (legacy support).
    Useful for comparison or when local compute is limited.
    """
    
    def __init__(self, region: str = "us-west-2"):
        self.region = region
        self._client = None
    
    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """
        For AWS Transcribe, we'd need to send audio to the streaming API.
        This is a simplified batch version.
        """
        # Placeholder - would need full AWS streaming implementation
        return TranscriptResult(
            text="[AWS Transcribe not implemented in v2]",
            is_final=True,
            confidence=0.0,
            timestamp=time.time()
        )
    
    async def transcribe_async(self, audio: np.ndarray) -> TranscriptResult:
        return self.transcribe(audio)
    
    def supports_streaming(self) -> bool:
        return True  # AWS Transcribe does support streaming


class Transcriber:
    """
    High-level transcription manager.
    Connects AudioPipeline to ASR engine and manages the flow.
    
    Priority: Parakeet TDT > faster-whisper > AWS Transcribe
    Falls back automatically if preferred engine fails to load.
    """
    
    def __init__(self, audio_pipeline: AudioPipeline):
        self.config = get_config()
        self.audio_pipeline = audio_pipeline
        
        # Create ASR engine based on config with fallback
        self.engine = self._create_engine()
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Output queue
        self._transcript_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # Callbacks
        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_final: Optional[Callable[[TranscriptResult], None]] = None
    
    def _create_engine(self) -> ASREngine:
        """Create ASR engine with automatic fallback"""
        backend = self.config.asr.backend
        
        # Try Parakeet TDT first if configured
        if backend == ASRBackend.PARAKEET_TDT:
            try:
                engine = ParakeetEngine()
                if engine.is_loaded:
                    print(f"[ASR] Using Parakeet TDT (streaming, SOTA)")
                    return engine
                else:
                    print(f"[ASR] Parakeet failed to load, falling back to faster-whisper")
                    backend = ASRBackend.FASTER_WHISPER
            except Exception as e:
                print(f"[ASR] Parakeet error: {e}, falling back to faster-whisper")
                backend = ASRBackend.FASTER_WHISPER
        
        # Try faster-whisper
        if backend == ASRBackend.FASTER_WHISPER:
            try:
                engine = FasterWhisperEngine(
                    model_size=self.config.asr.model_size,
                    device=self.config.asr.device,
                    compute_type=self.config.asr.compute_type,
                    language=self.config.asr.language,
                    beam_size=self.config.asr.beam_size
                )
                print(f"[ASR] Using faster-whisper ({self.config.asr.model_size})")
                return engine
            except Exception as e:
                print(f"[ASR] faster-whisper error: {e}")
                if backend != ASRBackend.AWS_TRANSCRIBE:
                    backend = ASRBackend.AWS_TRANSCRIBE
        
        # Fall back to AWS Transcribe
        if backend == ASRBackend.AWS_TRANSCRIBE:
            print(f"[ASR] Using AWS Transcribe (cloud)")
            return AWSTranscribeEngine(region=self.config.asr.aws_region)
        
        raise RuntimeError("No ASR engine could be loaded!")
    
    def set_partial_callback(self, callback: Callable[[str], None]):
        """Set callback for partial transcripts (for UI feedback)"""
        self._on_partial = callback
    
    def set_final_callback(self, callback: Callable[[TranscriptResult], None]):
        """Set callback for final transcripts"""
        self._on_final = callback
    
    async def start(self):
        """Start transcription processing"""
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        print("[TRANSCRIBER] Started")
    
    async def stop(self):
        """Stop transcription processing"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("[TRANSCRIBER] Stopped")
    
    async def _process_loop(self):
        """Main processing loop - transcribe speech segments"""
        async for segment in self.audio_pipeline.speech_segments():
            if not self._running:
                break
            
            # Show "listening" feedback
            if self._on_partial:
                self._on_partial("...")
            
            # Transcribe
            start = time.time()
            result = await self.engine.transcribe_async(segment.audio)
            latency = (time.time() - start) * 1000
            
            # Enrich result with timing info
            result.start_time = segment.start_time
            result.end_time = segment.end_time
            
            if self.config.debug:
                print(f"[ASR] \"{result.text}\" (latency={latency:.0f}ms, conf={result.confidence:.2f})")
            
            # Callbacks
            if self._on_final and result.text:
                self._on_final(result)
            
            # Queue for async consumers
            try:
                self._transcript_queue.put_nowait(result)
            except asyncio.QueueFull:
                pass
    
    async def transcripts(self) -> AsyncIterator[TranscriptResult]:
        """Async iterator yielding transcription results"""
        while self._running:
            try:
                result = await asyncio.wait_for(
                    self._transcript_queue.get(),
                    timeout=0.5
                )
                if result.text:  # Only yield non-empty results
                    yield result
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    def transcribe_audio(self, audio: np.ndarray) -> TranscriptResult:
        """Direct transcription (blocking) - for one-off use"""
        return self.engine.transcribe(audio)


def create_transcriber(audio_pipeline: AudioPipeline) -> Transcriber:
    """Factory function to create transcriber"""
    return Transcriber(audio_pipeline)
