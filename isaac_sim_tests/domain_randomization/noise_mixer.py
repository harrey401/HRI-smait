"""
SMAIT HRI v2.0 - Noise Mixer

Injects background noise into audio streams to test robustness.

Noise types:
- Ambient (cafe, street, office)
- Music
- Overlapping speech (hardest case)

Usage:
    mixer = NoiseMixer()
    mixer.load_noise("cafe_ambience.wav")
    noisy_audio = mixer.mix(clean_audio, noise_level=0.3)
"""

import numpy as np
import wave
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import threading


@dataclass
class NoiseProfile:
    """A loaded noise source"""
    name: str
    audio: np.ndarray
    sample_rate: int
    duration: float
    
    @property
    def samples(self) -> int:
        return len(self.audio)


class NoiseMixer:
    """
    Mixes background noise with speech audio.
    
    Supports multiple noise sources with different mixing levels.
    Uses circular playback for continuous noise.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Loaded noise sources
        self._noise_profiles: Dict[str, NoiseProfile] = {}
        
        # Current noise state
        self._active_noise: Optional[str] = None
        self._noise_level: float = 0.0
        self._noise_position: int = 0  # Current position in noise buffer
        
        self._lock = threading.Lock()
    
    def load_noise(self, audio_path: str, name: Optional[str] = None) -> bool:
        """
        Load a noise file.
        
        Args:
            audio_path: Path to WAV file
            name: Optional name for this noise (defaults to filename)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            print(f"[NOISE] File not found: {audio_path}")
            return False
        
        if name is None:
            name = audio_path.stem
        
        try:
            with wave.open(str(audio_path), 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                n_channels = wav.getnchannels()
                
                audio_bytes = wav.readframes(n_frames)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                audio /= 32768.0  # Normalize to [-1, 1]
                
                # Convert stereo to mono if needed
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                
                # Resample if needed
                if sample_rate != self.sample_rate:
                    audio = self._resample(audio, sample_rate, self.sample_rate)
                
                duration = len(audio) / self.sample_rate
                
                self._noise_profiles[name] = NoiseProfile(
                    name=name,
                    audio=audio,
                    sample_rate=self.sample_rate,
                    duration=duration
                )
                
                print(f"[NOISE] Loaded '{name}': {duration:.1f}s @ {self.sample_rate}Hz")
                return True
                
        except Exception as e:
            print(f"[NOISE] Failed to load {audio_path}: {e}")
            return False
    
    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Simple resampling (linear interpolation)"""
        if from_sr == to_sr:
            return audio
        
        ratio = to_sr / from_sr
        new_length = int(len(audio) * ratio)
        
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
    
    def set_active_noise(self, name: str, level: float = 0.3):
        """
        Set the active noise source and level.
        
        Args:
            name: Name of loaded noise profile
            level: Mix level (0.0 to 1.0, where 1.0 is equal volume to speech)
        """
        with self._lock:
            if name not in self._noise_profiles:
                print(f"[NOISE] Unknown noise profile: {name}")
                return
            
            self._active_noise = name
            self._noise_level = np.clip(level, 0.0, 1.0)
            self._noise_position = 0
            
            print(f"[NOISE] Active: {name} @ {level:.0%}")
    
    def clear_noise(self):
        """Disable noise injection"""
        with self._lock:
            self._active_noise = None
            self._noise_level = 0.0
    
    def mix(self, speech: np.ndarray) -> np.ndarray:
        """
        Mix noise with speech audio.
        
        Args:
            speech: Clean speech audio (float32, normalized to [-1, 1])
            
        Returns:
            Noisy audio (same shape as input)
        """
        with self._lock:
            if self._active_noise is None or self._noise_level == 0.0:
                return speech
            
            noise_profile = self._noise_profiles[self._active_noise]
            noise = noise_profile.audio
            
            # Get noise segment (circular buffer)
            speech_len = len(speech)
            noise_segment = np.zeros(speech_len)
            
            pos = self._noise_position
            remaining = speech_len
            write_pos = 0
            
            while remaining > 0:
                # How much can we copy from current position?
                available = len(noise) - pos
                to_copy = min(available, remaining)
                
                noise_segment[write_pos:write_pos + to_copy] = noise[pos:pos + to_copy]
                
                write_pos += to_copy
                pos = (pos + to_copy) % len(noise)
                remaining -= to_copy
            
            # Update position for next call
            self._noise_position = pos
            
            # Mix: speech + scaled noise
            mixed = speech + self._noise_level * noise_segment
            
            # Soft clipping to prevent harsh distortion
            mixed = np.tanh(mixed)
            
            return mixed
    
    def mix_with_snr(self, speech: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Mix noise with speech at a specific Signal-to-Noise Ratio.
        
        Args:
            speech: Clean speech audio
            snr_db: Signal-to-noise ratio in dB (e.g., 10 = speech 10dB louder)
        """
        with self._lock:
            if self._active_noise is None:
                return speech
            
            # Calculate speech power
            speech_power = np.mean(speech ** 2)
            
            # Get noise segment
            noise_profile = self._noise_profiles[self._active_noise]
            noise = noise_profile.audio[:len(speech)]
            
            if len(noise) < len(speech):
                # Repeat noise to match speech length
                repeats = int(np.ceil(len(speech) / len(noise)))
                noise = np.tile(noise, repeats)[:len(speech)]
            
            # Calculate required noise power for target SNR
            # SNR = 10 * log10(signal_power / noise_power)
            # noise_power = signal_power / 10^(SNR/10)
            target_noise_power = speech_power / (10 ** (snr_db / 10))
            
            # Current noise power
            noise_power = np.mean(noise ** 2) + 1e-10
            
            # Scale noise
            scale = np.sqrt(target_noise_power / noise_power)
            scaled_noise = noise * scale
            
            # Mix
            mixed = speech + scaled_noise
            mixed = np.tanh(mixed)  # Soft clip
            
            return mixed
    
    @property
    def available_noises(self) -> List[str]:
        """List of loaded noise profiles"""
        return list(self._noise_profiles.keys())
    
    def get_noise_info(self, name: str) -> Optional[Dict]:
        """Get info about a noise profile"""
        if name not in self._noise_profiles:
            return None
        
        profile = self._noise_profiles[name]
        return {
            'name': profile.name,
            'duration': profile.duration,
            'sample_rate': profile.sample_rate,
            'samples': profile.samples
        }


class MultiNoiseMixer(NoiseMixer):
    """
    Extended mixer supporting multiple simultaneous noise sources.
    
    Useful for complex scenarios like cafe (ambient + music + chatter).
    """
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__(sample_rate)
        
        # Multiple active noises with individual levels
        self._active_noises: Dict[str, float] = {}  # name -> level
        self._noise_positions: Dict[str, int] = {}
    
    def add_noise(self, name: str, level: float):
        """Add a noise source to the mix"""
        if name not in self._noise_profiles:
            print(f"[NOISE] Unknown noise profile: {name}")
            return
        
        self._active_noises[name] = np.clip(level, 0.0, 1.0)
        if name not in self._noise_positions:
            self._noise_positions[name] = 0
    
    def remove_noise(self, name: str):
        """Remove a noise source from the mix"""
        self._active_noises.pop(name, None)
    
    def mix(self, speech: np.ndarray) -> np.ndarray:
        """Mix all active noises with speech"""
        if not self._active_noises:
            return speech
        
        mixed = speech.copy()
        
        for name, level in self._active_noises.items():
            if level == 0.0:
                continue
            
            noise_profile = self._noise_profiles[name]
            noise = noise_profile.audio
            
            # Get noise segment (circular buffer)
            speech_len = len(speech)
            noise_segment = np.zeros(speech_len)
            
            pos = self._noise_positions.get(name, 0)
            remaining = speech_len
            write_pos = 0
            
            while remaining > 0:
                available = len(noise) - pos
                to_copy = min(available, remaining)
                noise_segment[write_pos:write_pos + to_copy] = noise[pos:pos + to_copy]
                write_pos += to_copy
                pos = (pos + to_copy) % len(noise)
                remaining -= to_copy
            
            self._noise_positions[name] = pos
            
            mixed += level * noise_segment
        
        # Soft clipping
        mixed = np.tanh(mixed)
        
        return mixed


def generate_test_noises(output_dir: str = "./audio_samples/noise"):
    """Generate synthetic test noise files"""
    from scipy.io import wavfile
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    duration = 30  # seconds
    n_samples = sample_rate * duration
    
    print(f"Generating test noise files in {output_dir}")
    
    # White noise (baseline)
    white_noise = np.random.randn(n_samples).astype(np.float32) * 0.1
    wavfile.write(output_dir / "white_noise.wav", sample_rate, 
                  (white_noise * 32767).astype(np.int16))
    print("  Created white_noise.wav")
    
    # Pink noise (more realistic ambient)
    # 1/f noise approximation
    freqs = np.fft.rfftfreq(n_samples, 1/sample_rate)
    freqs[0] = 1  # Avoid division by zero
    pink_spectrum = 1 / np.sqrt(freqs)
    pink_phases = np.random.uniform(0, 2*np.pi, len(freqs))
    pink_complex = pink_spectrum * np.exp(1j * pink_phases)
    pink_noise = np.fft.irfft(pink_complex, n_samples).astype(np.float32)
    pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.1
    wavfile.write(output_dir / "pink_noise.wav", sample_rate,
                  (pink_noise * 32767).astype(np.int16))
    print("  Created pink_noise.wav")
    
    # Babble (simulated overlapping speech)
    # Multiple copies of noise with different delays
    babble = np.zeros(n_samples)
    for i in range(8):  # 8 "voices"
        offset = np.random.randint(0, n_samples // 4)
        voice = np.random.randn(n_samples).astype(np.float32)
        # Simple formant simulation
        voice = np.convolve(voice, np.ones(100)/100, mode='same')
        babble += np.roll(voice, offset)
    babble = babble / np.max(np.abs(babble)) * 0.15
    wavfile.write(output_dir / "babble.wav", sample_rate,
                  (babble.astype(np.float32) * 32767).astype(np.int16))
    print("  Created babble.wav")
    
    print("Done!")


if __name__ == "__main__":
    # Generate test noises
    generate_test_noises()
    
    # Test the mixer
    print("\nTesting NoiseMixer...")
    mixer = NoiseMixer()
    
    # Load generated noise
    mixer.load_noise("./audio_samples/noise/pink_noise.wav", "ambient")
    
    # Create fake speech
    speech = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.5
    
    # Mix at different levels
    for level in [0.1, 0.3, 0.5]:
        mixer.set_active_noise("ambient", level)
        noisy = mixer.mix(speech)
        print(f"Level {level:.0%}: max={np.max(np.abs(noisy)):.3f}")
