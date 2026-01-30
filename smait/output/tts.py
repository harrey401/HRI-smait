"""
SMAIT HRI System v2.0 - Text-to-Speech Engine

Supports:
- Piper TTS (local, fast ~50ms) - RECOMMENDED
- Edge TTS (cloud, natural voice ~1200ms)
"""

import asyncio
import io
import time
import wave
import threading
from abc import ABC, abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path

from smait.core.config import get_config


@dataclass
class TTSResult:
    """Result from TTS synthesis"""
    audio_data: bytes  # Raw audio bytes
    duration_ms: float  # Estimated duration
    latency_ms: float  # Time to synthesize
    text: str  # Original text
    format: str = "mp3"  # Audio format: "mp3" or "wav"


class TTSEngine(ABC):
    """Abstract TTS engine interface"""

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize text to speech (async)"""
        pass

    @abstractmethod
    def synthesize_sync(self, text: str) -> TTSResult:
        """Synthesize text to speech (blocking)"""
        pass


class EdgeTTSEngine(TTSEngine):
    """
    Edge TTS engine using Microsoft Edge's online TTS service.

    Features:
    - Natural-sounding voices
    - Multiple languages and voices
    - Free to use
    - Fast synthesis
    """

    # Popular voice options
    VOICES = {
        # English (US)
        "en-us-female": "en-US-JennyNeural",
        "en-us-male": "en-US-GuyNeural",
        "en-us-aria": "en-US-AriaNeural",
        "en-us-davis": "en-US-DavisNeural",
        # English (UK)
        "en-gb-female": "en-GB-SoniaNeural",
        "en-gb-male": "en-GB-RyanNeural",
        # English (AU)
        "en-au-female": "en-AU-NatashaNeural",
        "en-au-male": "en-AU-WilliamNeural",
    }

    def __init__(
        self,
        voice: str = "en-us-aria",
        rate: str = "+0%",  # Speech rate adjustment (-50% to +100%)
        pitch: str = "+0Hz",  # Pitch adjustment (-50Hz to +50Hz)
        volume: str = "+0%",  # Volume adjustment (-50% to +50%)
    ):
        self.config = get_config()

        # Resolve voice name
        if voice in self.VOICES:
            self._voice = self.VOICES[voice]
        else:
            self._voice = voice  # Use as-is if not in presets

        self._rate = rate
        self._pitch = pitch
        self._volume = volume

        # For running async in sync context
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        print(f"[TTS] Edge TTS initialized (voice={self._voice})")

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize text to speech asynchronously"""
        import edge_tts

        start_time = time.time()

        # Create communicate object
        communicate = edge_tts.Communicate(
            text=text,
            voice=self._voice,
            rate=self._rate,
            pitch=self._pitch,
            volume=self._volume,
        )

        # Collect audio data
        audio_data = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])

        audio_bytes = audio_data.getvalue()
        latency = (time.time() - start_time) * 1000

        # Estimate duration (rough: ~10 chars per second for English)
        estimated_duration = len(text) * 100  # ms

        return TTSResult(
            audio_data=audio_bytes,
            duration_ms=estimated_duration,
            latency_ms=latency,
            text=text,
        )

    def synthesize_sync(self, text: str) -> TTSResult:
        """Synthesize text to speech (blocking wrapper)"""
        return asyncio.run(self.synthesize(text))

    def set_voice(self, voice: str):
        """Change voice"""
        if voice in self.VOICES:
            self._voice = self.VOICES[voice]
        else:
            self._voice = voice
        print(f"[TTS] Voice changed to {self._voice}")

    def set_rate(self, rate: str):
        """Change speech rate (e.g., '+10%', '-20%')"""
        self._rate = rate

    def set_pitch(self, pitch: str):
        """Change pitch (e.g., '+5Hz', '-10Hz')"""
        self._pitch = pitch

    @staticmethod
    async def list_voices(language: str = "en") -> list:
        """List available voices for a language"""
        import edge_tts

        voices = await edge_tts.list_voices()
        return [v for v in voices if v["Locale"].startswith(language)]


class PiperTTSEngine(TTSEngine):
    """
    Piper TTS engine - fast local neural TTS.

    Features:
    - Very fast (~50ms synthesis)
    - Runs locally (no internet needed)
    - Multiple voice models available
    - ONNX-based inference

    Requires: pip install piper-tts
    Models: https://huggingface.co/rhasspy/piper-voices
    """

    # Default model directory
    DEFAULT_MODEL_DIR = Path.home() / ".smait" / "models" / "piper"

    # Recommended voices (model name -> HuggingFace path)
    VOICES = {
        "amy": "en_US-amy-medium",
        "lessac": "en_US-lessac-medium",
        "libritts": "en_US-libritts-high",
        "ryan": "en_US-ryan-medium",
        "jenny": "en_GB-jenny_dioco-medium",
    }

    def __init__(
        self,
        voice: str = "amy",
        model_path: Optional[str] = None,
        use_cuda: bool = False,
    ):
        self.config = get_config()
        self._use_cuda = use_cuda
        self._voice = None
        self._model_path = None

        # Resolve model path
        if model_path:
            self._model_path = Path(model_path)
        else:
            # Use voice preset or default
            voice_name = self.VOICES.get(voice, voice)
            self._model_path = self.DEFAULT_MODEL_DIR / f"{voice_name}.onnx"

        # Try to load voice
        self._load_voice()

    def _load_voice(self):
        """Load the Piper voice model"""
        try:
            from piper.voice import PiperVoice

            if not self._model_path.exists():
                print(f"[TTS] Piper model not found at {self._model_path}")
                print(f"[TTS] Downloading model...")
                self._download_model()

            if self._model_path.exists():
                self._voice = PiperVoice.load(str(self._model_path), use_cuda=self._use_cuda)
                print(f"[TTS] Piper TTS initialized (model={self._model_path.stem})")
            else:
                print(f"[TTS] ERROR: Could not load Piper model")
                self._voice = None

        except ImportError:
            print("[TTS] ERROR: piper-tts not installed. Run: pip install piper-tts")
            self._voice = None
        except Exception as e:
            print(f"[TTS] ERROR loading Piper: {e}")
            self._voice = None

    def _download_model(self):
        """Download Piper voice model from HuggingFace"""
        try:
            import urllib.request

            # Create model directory
            self._model_path.parent.mkdir(parents=True, exist_ok=True)

            # Get model name from path
            model_name = self._model_path.stem

            # Parse model name to get language/region/voice
            # Format: en_US-amy-medium -> en/en_US/amy/medium/
            parts = model_name.split("-")
            if len(parts) >= 2:
                lang_region = parts[0]  # en_US
                lang = lang_region.split("_")[0]  # en
                voice = parts[1]  # amy
                quality = parts[2] if len(parts) > 2 else "medium"

                # HuggingFace URL pattern
                base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_region}/{voice}/{quality}"
                onnx_url = f"{base_url}/{model_name}.onnx"
                json_url = f"{base_url}/{model_name}.onnx.json"
            else:
                # Fallback for simple names
                onnx_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/{model_name}/{model_name}.onnx"
                json_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/{model_name}/{model_name}.onnx.json"

            print(f"[TTS] Downloading {model_name}.onnx (~50MB)...")
            urllib.request.urlretrieve(onnx_url, self._model_path)

            print(f"[TTS] Downloading {model_name}.onnx.json...")
            json_path = self._model_path.with_suffix(".onnx.json")
            urllib.request.urlretrieve(json_url, json_path)

            print(f"[TTS] Model downloaded to {self._model_path}")

        except Exception as e:
            print(f"[TTS] Failed to download model: {e}")
            print(f"[TTS] Please manually download from: https://huggingface.co/rhasspy/piper-voices")
            print(f"[TTS] Place .onnx and .onnx.json files in: {self._model_path.parent}")

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize text to speech asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize_sync, text)

    def synthesize_sync(self, text: str) -> TTSResult:
        """Synthesize text to speech (blocking)"""
        if self._voice is None:
            # Return empty result if voice not loaded
            return TTSResult(
                audio_data=b"",
                duration_ms=0,
                latency_ms=0,
                text=text,
                format="wav",
            )

        start_time = time.time()

        # Synthesize to WAV in memory
        audio_buffer = io.BytesIO()

        with wave.open(audio_buffer, "wb") as wav_file:
            # synthesize_wav sets the format automatically
            self._voice.synthesize_wav(text, wav_file)

        audio_data = audio_buffer.getvalue()
        latency = (time.time() - start_time) * 1000

        # Calculate duration from audio length
        # Default Piper sample rate is 22050 Hz, 16-bit mono = 44100 bytes/sec
        audio_bytes_only = len(audio_data) - 44  # Subtract WAV header
        duration_ms = (audio_bytes_only / 44100) * 1000

        return TTSResult(
            audio_data=audio_data,
            duration_ms=duration_ms,
            latency_ms=latency,
            text=text,
            format="wav",
        )

    def set_voice(self, voice: str):
        """Change voice model"""
        voice_name = self.VOICES.get(voice, voice)
        self._model_path = self.DEFAULT_MODEL_DIR / f"{voice_name}.onnx"
        self._load_voice()


class TTSPlayer:
    """
    Plays TTS audio output.
    Can play locally or send to remote (Android app).
    """

    def __init__(self, play_locally: bool = True):
        self._play_locally = play_locally
        self._audio_callback: Optional[Callable[[bytes], None]] = None

    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """Set callback to receive audio data (for sending to Android)"""
        self._audio_callback = callback

    def play(self, tts_result: TTSResult):
        """Play or send TTS audio"""
        if self._audio_callback:
            # Send to callback (e.g., Android app)
            self._audio_callback(tts_result.audio_data)

        if self._play_locally:
            self._play_local(tts_result.audio_data, tts_result.format)

    def _play_local(self, audio_data: bytes, audio_format: str = "mp3"):
        """Play audio locally using pygame or similar"""
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050 if audio_format == "wav" else 44100)

            # Load from bytes
            audio_io = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_io, audio_format)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except ImportError:
            # Fallback: save to temp file and play with system
            import tempfile
            import os

            suffix = f".{audio_format}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Try different players
                if os.name == "nt":  # Windows
                    os.system(f'start /min "" "{temp_path}"')
                else:  # Linux/Mac
                    os.system(f'mpv --no-video "{temp_path}" 2>/dev/null || afplay "{temp_path}" 2>/dev/null')
            finally:
                # Clean up after a delay
                threading.Timer(5.0, lambda: os.unlink(temp_path)).start()

    async def play_async(self, tts_result: TTSResult):
        """Play audio asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.play, tts_result)
