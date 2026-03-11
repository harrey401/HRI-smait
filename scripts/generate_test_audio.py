#!/usr/bin/env python3
"""Generate test audio fixtures for smoke tests and E2E testing.

Run at HOME:
    python scripts/generate_test_audio.py

Creates:
    scripts/test_audio/silence_16k.wav      - 2s silence
    scripts/test_audio/tone_16k.wav         - 2s 440Hz sine wave
    scripts/test_audio/speech_like_16k.wav  - 2s synthetic speech-like signal

For BEST results, also record yourself saying something and save as:
    scripts/test_audio/sample_16k.wav (16kHz mono WAV)

You can record with:
    arecord -f S16_LE -r 16000 -c 1 -d 5 scripts/test_audio/sample_16k.wav
"""

import os
import sys

import numpy as np


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Write a WAV file (minimal, no extra dependencies)."""
    import struct
    import wave

    # Ensure int16
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "test_audio")
    os.makedirs(out_dir, exist_ok=True)

    sr = 16000

    # 1. Silence
    silence = np.zeros(sr * 2, dtype=np.float32)
    path = os.path.join(out_dir, "silence_16k.wav")
    write_wav(path, silence, sr)
    print(f"Created: {path} (2s silence)")

    # 2. Pure tone
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = os.path.join(out_dir, "tone_16k.wav")
    write_wav(path, tone, sr)
    print(f"Created: {path} (2s 440Hz tone)")

    # 3. Speech-like signal (modulated noise with formant-like frequencies)
    # Simulate speech by mixing narrow-band noise at formant frequencies
    np.random.seed(42)
    noise = np.random.randn(sr * 2).astype(np.float32) * 0.3

    # Simple formant simulation: bandpass around 300Hz, 1000Hz, 2500Hz
    from scipy.signal import butter, lfilter

    def bandpass(data, low, high, sr, order=4):
        nyq = sr / 2
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return lfilter(b, a, data)

    try:
        f1 = bandpass(noise, 200, 400, sr) * 2.0
        f2 = bandpass(noise, 800, 1200, sr) * 1.5
        f3 = bandpass(noise, 2200, 2800, sr) * 1.0

        # Amplitude modulation to simulate syllable rhythm (~4 Hz)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        speech_like = (f1 + f2 + f3) * envelope
        speech_like = speech_like / np.max(np.abs(speech_like)) * 0.7
    except ImportError:
        # No scipy — use simple modulated noise
        speech_like = noise * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))

    path = os.path.join(out_dir, "speech_like_16k.wav")
    write_wav(path, speech_like, sr)
    print(f"Created: {path} (2s speech-like)")

    # 4. Multi-speaker simulation (two tones at different frequencies)
    speaker1 = 0.4 * np.sin(2 * np.pi * 200 * t)  # Low voice
    speaker2 = 0.4 * np.sin(2 * np.pi * 350 * t)  # Higher voice
    # Speaker 1 talks first half, both talk second half
    mixed = np.zeros_like(t)
    mixed[:sr] = speaker1[:sr]
    mixed[sr:] = speaker1[sr:] + speaker2[sr:]
    mixed = mixed / np.max(np.abs(mixed)) * 0.7

    path = os.path.join(out_dir, "multi_speaker_16k.wav")
    write_wav(path, mixed, sr)
    print(f"Created: {path} (2s multi-speaker)")

    print(f"\nAll test audio saved to {out_dir}/")
    print("\nFor best results, also record real speech:")
    print("  arecord -f S16_LE -r 16000 -c 1 -d 5 scripts/test_audio/sample_16k.wav")

    return 0


if __name__ == "__main__":
    sys.exit(main())
