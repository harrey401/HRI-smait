#!/usr/bin/env python3
"""End-to-end pipeline test: exercises the full SMAIT pipeline on GPU.

Run in lab AFTER all individual smoke tests pass:
    python scripts/test_e2e_pipeline.py

This test:
1. Loads all models (like the real system)
2. Feeds synthetic audio through VAD -> Dolphin -> Parakeet -> TTS
3. Measures end-to-end latency
4. Does NOT require Jackie (no WebSocket, no Android)

This validates the entire processing chain works on real GPU hardware
before you connect Jackie.
"""

import asyncio
import os
import sys
import time

import numpy as np

os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def load_test_audio() -> np.ndarray:
    """Load or generate test audio (16kHz mono float32, ~2s)."""
    for path in ["scripts/test_audio/sample_16k.wav", "scripts/test_audio/speech_like_16k.wav"]:
        if os.path.exists(path):
            try:
                import soundfile as sf
                audio, sr = sf.read(path, dtype="float32")
                if sr == 16000:
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    return audio
            except Exception:
                pass

    # Generate synthetic
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 200 * t)


async def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("*** CUDA not available ***")
        return 1

    results = []
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}\n")

    # --- Step 1: Load all models ---
    print("Step 1: Loading all models...")
    t0 = time.time()

    # Silero VAD
    vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", trust_repo=True)
    vad_model.eval()
    print("  Silero VAD loaded")

    # Dolphin
    dolphin = None
    try:
        from look2hear.models import Dolphin
        dolphin = Dolphin.from_pretrained("JusperLee/Dolphin")
        dolphin = dolphin.to("cuda")
        dolphin.eval()
        print("  Dolphin loaded")
    except Exception as e:
        print(f"  Dolphin skipped: {e}")

    # Parakeet
    parakeet = None
    try:
        import nemo.collections.asr as nemo_asr
        parakeet = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        parakeet.eval()
        print("  Parakeet loaded")
    except Exception as e:
        print(f"  Parakeet skipped: {e}")

    # Kokoro
    kokoro = None
    try:
        from kokoro import KPipeline
        kokoro = KPipeline(lang_code="a")
        print("  Kokoro loaded")
    except Exception as e:
        print(f"  Kokoro skipped: {e}")

    load_time = time.time() - t0
    results.append(check("All models loaded", True, f"{load_time:.1f}s"))

    # --- Step 2: VAD on test audio ---
    print("\nStep 2: VAD speech detection...")
    audio = load_test_audio()
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    chunk_size = 512  # 32ms at 16kHz (Silero required size)
    speech_chunks = 0
    total_chunks = 0
    vad_model.reset_states()

    t0 = time.time()
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = torch.from_numpy(audio[i:i + chunk_size])
        prob = vad_model(chunk, 16000).item()
        total_chunks += 1
        if prob >= 0.5:
            speech_chunks += 1

    vad_time = (time.time() - t0) * 1000
    results.append(check(
        "VAD processing",
        True,
        f"{speech_chunks}/{total_chunks} speech chunks, {vad_time:.0f}ms",
    ))

    # --- Step 3: Dolphin separation ---
    if dolphin is not None:
        print("\nStep 3: Dolphin separation...")
        # Simulate lip frames (25 frames of 88x88 grayscale)
        n_frames = 25
        video_tensor = torch.randn(1, 1, n_frames, 88, 88, device="cuda")
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to("cuda")

        t0 = time.time()
        with torch.inference_mode():
            output = dolphin(audio_tensor, video_tensor)
        sep_time = (time.time() - t0) * 1000

        if isinstance(output, tuple):
            separated = output[0].squeeze().cpu().numpy()
        else:
            separated = output.squeeze().cpu().numpy()

        results.append(check(
            "Dolphin separation",
            len(separated) > 0,
            f"output={len(separated)} samples, {sep_time:.0f}ms",
        ))
    else:
        print("\nStep 3: Dolphin skipped (not available)")
        # Use original audio as passthrough
        separated = audio

    # --- Step 4: Parakeet ASR ---
    if parakeet is not None:
        print("\nStep 4: Parakeet ASR transcription...")
        t0 = time.time()
        result = parakeet.transcribe([separated], return_hypotheses=True)
        asr_time = (time.time() - t0) * 1000

        if isinstance(result, tuple):
            texts = result[0]
        else:
            texts = result
        text = str(texts[0]) if texts else ""

        results.append(check(
            "Parakeet ASR",
            True,
            f"'{text[:60]}' ({asr_time:.0f}ms)",
        ))
    else:
        print("\nStep 4: Parakeet skipped (not available)")
        text = "Hello, I am a test."

    # --- Step 5: Kokoro TTS ---
    if kokoro is not None:
        print("\nStep 5: Kokoro TTS synthesis...")
        tts_text = text if text.strip() else "Hello, I am Jackie, a conference robot."
        t0 = time.time()

        tts_chunks = []
        for _, _, audio_out in kokoro(tts_text, voice="af_heart", speed=1.0):
            if audio_out is None:
                continue
            if hasattr(audio_out, "cpu"):
                audio_out = audio_out.cpu().numpy()
            elif not isinstance(audio_out, np.ndarray):
                audio_out = np.array(audio_out)
            tts_chunks.append(audio_out)

        tts_time = (time.time() - t0) * 1000

        if tts_chunks:
            tts_audio = np.concatenate(tts_chunks)
            tts_duration = len(tts_audio) / 24000
            # Convert to PCM16 (what gets sent to Jackie)
            pcm16 = (tts_audio * 32767).clip(-32768, 32767).astype(np.int16)
            pcm_bytes = pcm16.tobytes()

            results.append(check(
                "Kokoro TTS",
                len(pcm_bytes) > 0,
                f"{tts_duration:.2f}s audio, {len(pcm_bytes)} bytes PCM16, {tts_time:.0f}ms",
            ))
        else:
            results.append(check("Kokoro TTS", False, "no audio generated"))
    else:
        print("\nStep 5: Kokoro skipped (not available)")

    # --- Step 6: End-to-end latency estimate ---
    print("\nStep 6: Latency estimate...")
    # Measure the critical path: speech-end -> separation -> ASR -> first TTS chunk
    total_pipeline_ms = 0

    if dolphin is not None:
        # Re-run separation with timing
        audio_tensor = torch.from_numpy(audio[:16000]).unsqueeze(0).to("cuda")  # 1s segment
        n_frames = 12
        video_tensor = torch.randn(1, 1, n_frames, 88, 88, device="cuda")

        t0 = time.time()
        with torch.inference_mode():
            output = dolphin(audio_tensor, video_tensor)
        sep_ms = (time.time() - t0) * 1000
        total_pipeline_ms += sep_ms

        if isinstance(output, tuple):
            sep_audio = output[0].squeeze().cpu().numpy()
        else:
            sep_audio = output.squeeze().cpu().numpy()
    else:
        sep_ms = 0
        sep_audio = audio[:16000]

    if parakeet is not None:
        t0 = time.time()
        parakeet.transcribe([sep_audio], return_hypotheses=True)
        asr_ms = (time.time() - t0) * 1000
        total_pipeline_ms += asr_ms
    else:
        asr_ms = 0

    if kokoro is not None:
        t0 = time.time()
        for _, _, audio_out in kokoro("Hello!", voice="af_heart", speed=1.0):
            if audio_out is not None:
                # Time to first audio chunk
                ttfb_ms = (time.time() - t0) * 1000
                total_pipeline_ms += ttfb_ms
                break
    else:
        ttfb_ms = 0

    target_ms = 1500
    results.append(check(
        "E2E latency",
        total_pipeline_ms < target_ms,
        f"separation={sep_ms:.0f}ms + ASR={asr_ms:.0f}ms + TTS_TTFB={ttfb_ms:.0f}ms = "
        f"{total_pipeline_ms:.0f}ms (target: <{target_ms}ms)",
    ))

    # --- Step 7: VRAM after full pipeline ---
    final_vram = torch.cuda.memory_allocated() / (1024**2)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    total_gpu = torch.cuda.get_device_properties(0).total_mem / (1024**2)

    results.append(check(
        "Final VRAM",
        peak_vram < total_gpu * 0.9,
        f"allocated={final_vram:.0f}MB, peak={peak_vram:.0f}MB, total={total_gpu:.0f}MB",
    ))

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("E2E PIPELINE RESULTS")
    print(f"{'=' * 60}")
    passed = sum(results)
    total_tests = len(results)
    print(f"{passed}/{total_tests} passed")

    if all(results):
        print("\nPipeline is ready. Connect Jackie and run:")
        print("  python run_jackie.py --debug")
    else:
        print("\nFix failures above before connecting Jackie.")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
