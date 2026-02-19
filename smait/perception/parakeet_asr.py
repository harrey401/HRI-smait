"""
SMAIT HRI System v2.0 - Parakeet TDT ASR Engine
NVIDIA NeMo-based streaming speech recognition using Parakeet TDT.

This is the RECOMMENDED ASR backend for lowest latency.
- Model: nvidia/parakeet-tdt-0.6b-v2 (CC-BY-4.0, open source)
- Architecture: FastConformer + TDT (Token-and-Duration Transducer)
- Features: Streaming capable, punctuation, capitalization, word timestamps
- Performance: ~3000x RTFx on GPU, sub-100ms latency for streaming

References:
- HuggingFace: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- Paper: Fast Conformer with Linearly Scalable Attention
"""

import asyncio
import threading
import time
from typing import Optional, AsyncIterator, Generator, List, Dict, Any
from collections import deque
import numpy as np

from smait.core.config import get_config
from smait.core.events import TranscriptResult


class ParakeetTDTEngine:
    """
    Parakeet TDT ASR engine using NVIDIA NeMo.
    
    Supports two modes:
    1. Batch transcription: Process complete audio segments
    2. Streaming transcription: Process audio chunks incrementally
    
    The streaming mode uses NeMo's chunked inference for low-latency
    partial results while maintaining accuracy.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v2",
        device: str = "auto",
        streaming: bool = True,
        chunk_seconds: float = 2.0
    ):
        self.model_name = model_name
        self.device = device
        self.streaming = streaming
        self.chunk_seconds = chunk_seconds
        
        self.model = None
        self._lock = threading.Lock()
        self._loaded = False
        
        # Streaming state
        self._audio_buffer = deque(maxlen=int(16000 * 30))  # 30 sec buffer
        self._partial_text = ""
        
        self._load_model()
    
    def _load_model(self):
        """Load the Parakeet TDT model from HuggingFace via NeMo"""
        try:
            import torch
            import nemo.collections.asr as nemo_asr
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            print(f"[ASR] Loading Parakeet TDT ({self.model_name}) on {device}...")
            print(f"[ASR] First load will download model from HuggingFace (~1.2GB)")
            
            # Load model from HuggingFace
            # NeMo automatically handles caching
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            
            # Move to device with half precision to save VRAM
            self.model = self.model.to(device)
            if device == "cuda":
                self.model = self.model.half()  # fp16 — cuts VRAM ~50%
            self.model.eval()

            # Disable CUDA graph decoder — breaks on Blackwell (RTX 50xx) GPUs
            # due to changed CUDA API return format in CUDA 12.x+
            try:
                from omegaconf import OmegaConf, open_dict
                with open_dict(self.model.cfg):
                    self.model.cfg.decoding.greedy.use_cuda_graph_decoder = False
                self.model.change_decoding_strategy(self.model.cfg.decoding)
                print(f"[ASR] CUDA graph decoder disabled (Blackwell compat)")
            except Exception as cg_err:
                print(f"[ASR] Could not disable CUDA graphs: {cg_err}")
            
            # Disable gradient computation for inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            self._loaded = True
            self._device = device
            
            print(f"[ASR] Parakeet TDT loaded successfully")
            print(f"[ASR] Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
            
        except ImportError as e:
            print(f"[ASR] NeMo not installed! Install with: pip install nemo_toolkit[asr]")
            print(f"[ASR] Error: {e}")
            raise
        except Exception as e:
            print(f"[ASR] Failed to load Parakeet TDT: {e}")
            print(f"[ASR] Will fall back to faster-whisper if available")
            raise
    
    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """
        Transcribe audio segment (batch mode).
        
        Args:
            audio: Audio samples (int16 or float32 at 16kHz)
        
        Returns:
            TranscriptResult with transcription
        """
        if not self._loaded:
            return TranscriptResult(
                text="[ASR model not loaded]",
                is_final=True,
                confidence=0.0,
                timestamp=time.time()
            )
        
        start_time = time.time()
        
        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)
        
        with self._lock:
            try:
                import torch
                
                # NeMo expects audio as a list of numpy arrays or file paths
                # For in-memory audio, we use transcribe() with audio arrays
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
                audio_length = torch.tensor([audio_float.shape[0]])
                
                # Move to device
                audio_tensor = audio_tensor.to(self._device)
                audio_length = audio_length.to(self._device)
                
                # Transcribe
                with torch.no_grad():
                    # Use the model's transcribe method for simplicity
                    # This handles all preprocessing internally
                    result = self.model.transcribe(
                        [audio_float],
                        batch_size=1,
                        return_hypotheses=True
                    )
                    # Newer NeMo (2.0+) returns (hypotheses, all_hypotheses) tuple
                    if isinstance(result, tuple):
                        hypotheses = result[0]
                    else:
                        hypotheses = result
                
                # Extract results
                if hypotheses and len(hypotheses) > 0:
                    hyp = hypotheses[0]
                    
                    # Handle both string and Hypothesis object returns
                    if isinstance(hyp, str):
                        text = hyp
                        confidence = 0.9  # Default confidence for string returns
                        words = []
                    else:
                        text = hyp.text if hasattr(hyp, 'text') else str(hyp)
                        confidence = float(hyp.score) if hasattr(hyp, 'score') else 0.9
                        
                        # Extract word timestamps if available
                        words = []
                        if hasattr(hyp, 'timestep') and hyp.timestep:
                            # TDT models provide duration info
                            pass  # Word extraction would go here
                else:
                    text = ""
                    confidence = 0.0
                    words = []
                
                latency = (time.time() - start_time) * 1000
                
                return TranscriptResult(
                    text=text.strip(),
                    is_final=True,
                    confidence=confidence,
                    timestamp=time.time(),
                    words=words
                )
                
            except Exception as e:
                print(f"[ASR] Parakeet transcription error: {e}")
                import traceback
                traceback.print_exc()
                
                return TranscriptResult(
                    text="",
                    is_final=True,
                    confidence=0.0,
                    timestamp=time.time()
                )
    
    async def transcribe_async(self, audio: np.ndarray) -> TranscriptResult:
        """Async wrapper for batch transcription"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio)
    
    def transcribe_streaming(
        self,
        audio_chunk: np.ndarray
    ) -> Generator[TranscriptResult, None, None]:
        """
        Streaming transcription - process audio incrementally.
        
        This uses NeMo's chunked inference to provide partial results
        with low latency while maintaining accuracy.
        
        Args:
            audio_chunk: New audio samples to process
        
        Yields:
            TranscriptResult objects (partial and final)
        """
        if not self._loaded:
            yield TranscriptResult(
                text="[ASR model not loaded]",
                is_final=True,
                confidence=0.0,
                timestamp=time.time()
            )
            return
        
        # Convert to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)
        
        # Add to buffer
        self._audio_buffer.extend(audio_float)
        
        # Check if we have enough audio for a chunk
        chunk_samples = int(self.chunk_seconds * 16000)
        
        if len(self._audio_buffer) >= chunk_samples:
            # Extract chunk from buffer
            chunk = np.array(list(self._audio_buffer)[:chunk_samples])
            
            # Transcribe chunk
            result = self.transcribe(chunk)
            
            if result.text:
                # Yield partial result
                yield TranscriptResult(
                    text=result.text,
                    is_final=False,  # Partial
                    confidence=result.confidence,
                    timestamp=time.time()
                )
                
                self._partial_text = result.text
            
            # Slide buffer (keep some overlap for context)
            overlap_samples = int(0.5 * 16000)  # 0.5 second overlap
            for _ in range(chunk_samples - overlap_samples):
                if self._audio_buffer:
                    self._audio_buffer.popleft()
    
    def finalize_streaming(self) -> TranscriptResult:
        """
        Finalize streaming transcription.
        Process any remaining audio in the buffer.
        
        Returns:
            Final TranscriptResult
        """
        if len(self._audio_buffer) > 0:
            # Process remaining audio
            remaining = np.array(list(self._audio_buffer))
            result = self.transcribe(remaining)
            
            # Clear buffer
            self._audio_buffer.clear()
            self._partial_text = ""
            
            return TranscriptResult(
                text=result.text,
                is_final=True,
                confidence=result.confidence,
                timestamp=time.time()
            )
        
        return TranscriptResult(
            text=self._partial_text,
            is_final=True,
            confidence=0.9,
            timestamp=time.time()
        )
    
    def reset_streaming(self):
        """Reset streaming state"""
        self._audio_buffer.clear()
        self._partial_text = ""
    
    def supports_streaming(self) -> bool:
        """Parakeet TDT supports streaming via chunked inference"""
        return True
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded


class ParakeetStreamingTranscriber:
    """
    High-level streaming transcriber using Parakeet TDT.
    
    Provides an async interface for real-time speech recognition
    with partial results for responsive UI feedback.
    """
    
    def __init__(self, engine: ParakeetTDTEngine):
        self.engine = engine
        self._running = False
        self._audio_queue: asyncio.Queue = None
        self._result_queue: asyncio.Queue = None
    
    async def start(self):
        """Start the streaming transcriber"""
        self._running = True
        self._audio_queue = asyncio.Queue(maxsize=100)
        self._result_queue = asyncio.Queue(maxsize=100)
        
        # Start processing task
        asyncio.create_task(self._process_loop())
    
    async def stop(self):
        """Stop the streaming transcriber"""
        self._running = False
        
        # Finalize any remaining audio
        final = self.engine.finalize_streaming()
        if final.text:
            await self._result_queue.put(final)
    
    async def feed_audio(self, audio: np.ndarray):
        """Feed audio chunk to the transcriber"""
        if self._running and self._audio_queue:
            await self._audio_queue.put(audio)
    
    async def _process_loop(self):
        """Process audio chunks and emit results"""
        while self._running:
            try:
                # Get audio chunk
                audio = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.1
                )
                
                # Process through engine (in thread pool)
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: list(self.engine.transcribe_streaming(audio))
                )
                
                # Emit results
                for result in results:
                    await self._result_queue.put(result)
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ASR] Streaming error: {e}")
    
    async def results(self) -> AsyncIterator[TranscriptResult]:
        """Async iterator for transcription results"""
        while self._running:
            try:
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=0.5
                )
                yield result
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


def create_parakeet_engine(config=None) -> ParakeetTDTEngine:
    """Factory function to create Parakeet TDT engine"""
    if config is None:
        config = get_config()
    
    return ParakeetTDTEngine(
        model_name=config.asr.parakeet_model,
        device=config.asr.device,
        streaming=config.asr.parakeet_streaming,
        chunk_seconds=config.asr.parakeet_chunk_seconds
    )
