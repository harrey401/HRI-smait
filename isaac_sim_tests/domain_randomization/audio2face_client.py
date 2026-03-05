"""
SMAIT HRI v2.0 - Audio2Face Client

Controls NVIDIA Audio2Face to generate realistic mouth movements
for testing the ASD system.

Audio2Face provides:
- Lip sync from audio input
- Blendshape output for facial animation
- Streaming API for real-time control

Usage:
    client = Audio2FaceClient()
    client.connect()
    client.play_audio("speech.wav")
    client.set_blendshape_scale(0.5)  # Reduce mouth movement
"""

import socket
import struct
import json
import time
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class A2FStatus:
    """Audio2Face connection status"""
    connected: bool = False
    character_loaded: bool = False
    is_playing: bool = False
    blendshape_scale: float = 1.0
    current_audio: Optional[str] = None


class Audio2FaceClient:
    """
    Client for NVIDIA Audio2Face streaming API.
    
    Audio2Face exposes a gRPC/HTTP API for:
    - Loading audio files
    - Streaming audio in real-time
    - Getting blendshape values
    - Controlling animation parameters
    """
    
    # Default Audio2Face ports
    DEFAULT_STREAMING_PORT = 12030
    DEFAULT_HTTP_PORT = 8011
    
    def __init__(
        self,
        host: str = "localhost",
        streaming_port: int = DEFAULT_STREAMING_PORT,
        http_port: int = DEFAULT_HTTP_PORT
    ):
        self.host = host
        self.streaming_port = streaming_port
        self.http_port = http_port
        
        self.status = A2FStatus()
        self._socket: Optional[socket.socket] = None
        
        # Blendshape mapping (ARKit compatible)
        self.blendshape_names = [
            "eyeBlinkLeft", "eyeBlinkRight",
            "jawOpen", "jawForward", "jawLeft", "jawRight",
            "mouthClose", "mouthFunnel", "mouthPucker",
            "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight",
            "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
            "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
            # ... more blendshapes
        ]
    
    def connect(self) -> bool:
        """Connect to Audio2Face"""
        try:
            # Try HTTP health check first
            import requests
            response = requests.get(
                f"http://{self.host}:{self.http_port}/A2F/GetInstances",
                timeout=5
            )
            
            if response.status_code == 200:
                self.status.connected = True
                print(f"[A2F] Connected to Audio2Face at {self.host}:{self.http_port}")
                return True
            
        except ImportError:
            print("[A2F] requests library not available, using socket")
        except Exception as e:
            print(f"[A2F] HTTP connection failed: {e}")
        
        # Fallback to socket connection
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5)
            self._socket.connect((self.host, self.streaming_port))
            self.status.connected = True
            print(f"[A2F] Connected via socket to {self.host}:{self.streaming_port}")
            return True
            
        except Exception as e:
            print(f"[A2F] Socket connection failed: {e}")
            self.status.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Audio2Face"""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
        
        self.status.connected = False
        print("[A2F] Disconnected")
    
    def load_character(self, character: str = "mark") -> bool:
        """
        Load a character in Audio2Face.
        
        Default characters: mark, claire
        """
        try:
            response = self._http_post("/A2F/LoadUSD", {
                "usd_path": f"omniverse://localhost/NVIDIA/Assets/Audio2Face/Samples/{character}.usd"
            })
            
            if response and response.get("success"):
                self.status.character_loaded = True
                print(f"[A2F] Character '{character}' loaded")
                return True
            
        except Exception as e:
            print(f"[A2F] Failed to load character: {e}")
        
        return False
    
    def play_audio(self, audio_path: str) -> bool:
        """
        Play audio file through Audio2Face.
        
        Args:
            audio_path: Path to WAV file (16kHz, mono recommended)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            print(f"[A2F] Audio file not found: {audio_path}")
            return False
        
        try:
            # Load audio data
            with wave.open(str(audio_path), 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)
            
            # Convert to float32
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array /= 32768.0  # Normalize
            
            # Send to Audio2Face
            response = self._http_post("/A2F/A2E/GenerateEmotionKeys", {
                "audio_data": audio_array.tolist(),
                "sample_rate": sample_rate
            })
            
            self.status.is_playing = True
            self.status.current_audio = str(audio_path)
            
            # Estimate playback duration
            duration = n_frames / sample_rate
            print(f"[A2F] Playing audio: {audio_path.name} ({duration:.2f}s)")
            
            return True
            
        except Exception as e:
            print(f"[A2F] Failed to play audio: {e}")
            return False
    
    def stream_audio(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Stream audio data directly to Audio2Face.
        
        Args:
            audio_data: Float32 audio samples, normalized to [-1, 1]
            sample_rate: Sample rate in Hz
        """
        if not self._socket:
            print("[A2F] Not connected via socket")
            return False
        
        try:
            # Pack audio data
            # Format: [4 bytes: num_samples][4 bytes: sample_rate][n*4 bytes: float32 samples]
            header = struct.pack('<II', len(audio_data), sample_rate)
            samples = audio_data.astype(np.float32).tobytes()
            
            self._socket.sendall(header + samples)
            return True
            
        except Exception as e:
            print(f"[A2F] Stream error: {e}")
            return False
    
    def set_blendshape_scale(self, scale: float):
        """
        Scale blendshape output (for articulation variation).
        
        Args:
            scale: 0.0 to 2.0 (1.0 = normal, <1.0 = lazy, >1.0 = exaggerated)
        """
        self.status.blendshape_scale = scale
        
        try:
            response = self._http_post("/A2F/SetBlendshapeMultiplier", {
                "multiplier": scale
            })
            print(f"[A2F] Blendshape scale set to {scale}")
            return True
        except:
            print(f"[A2F] Could not set blendshape scale (using local scaling)")
            return False
    
    def get_blendshapes(self) -> Optional[Dict[str, float]]:
        """Get current blendshape values"""
        try:
            response = self._http_get("/A2F/GetBlendshapes")
            
            if response:
                # Apply local scaling
                scaled = {}
                for name, value in response.items():
                    scaled[name] = value * self.status.blendshape_scale
                return scaled
                
        except Exception as e:
            print(f"[A2F] Failed to get blendshapes: {e}")
        
        return None
    
    def get_current_frame(self) -> Optional[Dict[str, Any]]:
        """
        Get current animation frame data.
        
        Returns dict with:
        - blendshapes: Dict of blendshape values
        - timestamp: Frame timestamp
        - is_speaking: Whether audio is currently playing
        """
        blendshapes = self.get_blendshapes()
        
        if blendshapes:
            # Determine if "speaking" based on jaw/mouth blendshapes
            jaw_open = blendshapes.get("jawOpen", 0.0)
            mouth_open = blendshapes.get("mouthOpen", jaw_open)
            
            is_speaking = mouth_open > 0.1 or jaw_open > 0.1
            
            return {
                "blendshapes": blendshapes,
                "timestamp": time.time(),
                "is_speaking": is_speaking,
                "mouth_openness": max(jaw_open, mouth_open)
            }
        
        return None
    
    def _http_get(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP GET request to Audio2Face"""
        try:
            import requests
            response = requests.get(
                f"http://{self.host}:{self.http_port}{endpoint}",
                timeout=5
            )
            return response.json()
        except:
            return None
    
    def _http_post(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Make HTTP POST request to Audio2Face"""
        try:
            import requests
            response = requests.post(
                f"http://{self.host}:{self.http_port}{endpoint}",
                json=data,
                timeout=30
            )
            return response.json()
        except:
            return None


class Audio2FaceGroundTruth:
    """
    Provides ground truth speaking labels from Audio2Face.
    
    This tells us WHEN the simulated face is actually speaking,
    which we compare against our ASD predictions.
    """
    
    def __init__(self, client: Audio2FaceClient):
        self.client = client
        self.speaking_threshold = 0.1  # Mouth openness threshold
        
        # History for temporal analysis
        self.history: list = []
        self.history_duration = 2.0  # seconds
    
    def is_speaking(self) -> bool:
        """Check if the character is currently speaking"""
        frame = self.client.get_current_frame()
        
        if frame:
            is_speaking = frame['mouth_openness'] > self.speaking_threshold
            
            # Add to history
            self.history.append({
                'timestamp': frame['timestamp'],
                'speaking': is_speaking,
                'openness': frame['mouth_openness']
            })
            
            # Trim old history
            cutoff = time.time() - self.history_duration
            self.history = [h for h in self.history if h['timestamp'] > cutoff]
            
            return is_speaking
        
        return False
    
    def was_speaking_in_window(self, start_time: float, end_time: float) -> bool:
        """Check if character was speaking during a time window"""
        relevant = [
            h for h in self.history 
            if start_time <= h['timestamp'] <= end_time
        ]
        
        if not relevant:
            return False
        
        # Speaking if any frame in window had speaking
        return any(h['speaking'] for h in relevant)
    
    def get_speaking_ratio(self, start_time: float, end_time: float) -> float:
        """Get ratio of time spent speaking in window"""
        relevant = [
            h for h in self.history 
            if start_time <= h['timestamp'] <= end_time
        ]
        
        if not relevant:
            return 0.0
        
        speaking_count = sum(1 for h in relevant if h['speaking'])
        return speaking_count / len(relevant)


# Convenience function
def create_a2f_test_setup(host: str = "localhost") -> tuple:
    """
    Create Audio2Face client and ground truth tracker.
    
    Returns:
        (client, ground_truth)
    """
    client = Audio2FaceClient(host=host)
    ground_truth = Audio2FaceGroundTruth(client)
    return client, ground_truth


if __name__ == "__main__":
    # Test connection
    client = Audio2FaceClient()
    
    if client.connect():
        print("Audio2Face connected!")
        
        # Try to get blendshapes
        bs = client.get_blendshapes()
        if bs:
            print(f"Got {len(bs)} blendshapes")
        
        client.disconnect()
    else:
        print("Could not connect to Audio2Face")
        print("Make sure Audio2Face is running with streaming enabled")
