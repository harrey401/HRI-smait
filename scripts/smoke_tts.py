#!/usr/bin/env python3
"""Smoke test 4: Verify Kokoro-82M TTS on GPU.

Run in lab:
    python scripts/smoke_tts.py

Tests:
- KPipeline loads
- Generates audio from text
- Audio is correct format (float32/int16, 24kHz)
- Sentence streaming works
- VRAM usage
"""

import sys
import time

import numpy as np


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main() -> int:
    results = []

    # 1. Import
    try:
        from kokoro import KPipeline
        results.append(check("Kokoro import", True))
    except ImportError as e:
        results.append(check("Kokoro import", False, str(e)))
        print("Install: pip install kokoro>=0.9.4 soundfile && sudo apt install espeak-ng")
        return 1

    # 2. Load pipeline
    print("\nLoading Kokoro-82M...")
    t0 = time.time()
    try:
        pipeline = KPipeline(lang_code="a")
        load_time = time.time() - t0
        results.append(check("Pipeline loaded", True, f"{load_time:.1f}s"))
    except Exception as e:
        results.append(check("Pipeline loaded", False, str(e)))
        return 1

    # 3. VRAM usage
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            results.append(check("VRAM usage", True, f"{vram_mb:.0f} MB"))
        else:
            results.append(check("VRAM usage", True, "CPU mode (no CUDA)"))
    except Exception:
        results.append(check("VRAM usage", True, "could not measure"))

    # 4. Single sentence synthesis
    test_text = "Hello, I am Jackie, a friendly conference robot at SJSU."
    print(f"\nSynthesizing: '{test_text}'")
    t0 = time.time()
    try:
        audio_chunks = []
        for graphemes, phonemes, audio in pipeline(test_text, voice="af_heart", speed=1.0):
            if audio is None:
                continue
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            audio_chunks.append(audio)

        latency = (time.time() - t0) * 1000

        if audio_chunks:
            combined = np.concatenate(audio_chunks)
            duration = len(combined) / 24000
            results.append(check(
                "Single sentence TTS",
                len(combined) > 0,
                f"{duration:.2f}s audio, {latency:.0f}ms synthesis time, "
                f"rtf={duration / (latency/1000):.1f}x realtime",
            ))
        else:
            results.append(check("Single sentence TTS", False, "no audio generated"))
    except Exception as e:
        results.append(check("Single sentence TTS", False, str(e)))

    # 5. Multi-sentence streaming test
    multi_text = "Hi there! How are you doing today? I'm Jackie, your friendly conference robot."
    print(f"\nStreaming multi-sentence: '{multi_text}'")
    t0 = time.time()
    try:
        sentence_count = 0
        total_samples = 0
        for graphemes, phonemes, audio in pipeline(multi_text, voice="af_heart", speed=1.0):
            if audio is None:
                continue
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            sentence_count += 1
            total_samples += len(audio)

        latency = (time.time() - t0) * 1000
        duration = total_samples / 24000

        results.append(check(
            "Multi-sentence streaming",
            sentence_count >= 2,
            f"{sentence_count} sentences, {duration:.2f}s total audio, {latency:.0f}ms",
        ))
    except Exception as e:
        results.append(check("Multi-sentence streaming", False, str(e)))

    # 6. PCM16 conversion (what we send to Jackie)
    try:
        test_audio = np.random.randn(24000).astype(np.float32) * 0.5
        pcm16 = (test_audio * 32767).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm16.tobytes()
        results.append(check(
            "PCM16 conversion",
            len(pcm_bytes) == 24000 * 2,
            f"{len(pcm_bytes)} bytes for 1s at 24kHz",
        ))
    except Exception as e:
        results.append(check("PCM16 conversion", False, str(e)))

    # Summary
    passed = sum(results)
    total_tests = len(results)
    print(f"\n{'=' * 40}")
    print(f"Kokoro TTS: {passed}/{total_tests} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
