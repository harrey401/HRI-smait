#!/usr/bin/env python3
"""Smoke test 2: Verify Parakeet TDT 0.6B v2 ASR on GPU.

Run in lab:
    python scripts/smoke_parakeet.py

Requires: nemo_toolkit[asr], test audio file.
"""

import os
import sys
import time

import numpy as np

# Disable CUDA graphs for Blackwell (sm_120)
os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def generate_test_audio() -> np.ndarray:
    """Generate a simple sine wave as test audio (1 second, 16kHz)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    # 440 Hz sine wave (not speech, but tests model loading + inference path)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


def load_test_audio() -> np.ndarray:
    """Load real test audio if available, else generate synthetic."""
    test_files = [
        "scripts/test_audio/sample_16k.wav",
        "scripts/test_audio/hello.wav",
    ]
    for path in test_files:
        if os.path.exists(path):
            try:
                import soundfile as sf
                audio, sr = sf.read(path, dtype="float32")
                if sr != 16000:
                    print(f"  Warning: {path} is {sr}Hz, need 16kHz. Using synthetic.")
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                print(f"  Using real audio: {path} ({len(audio)/16000:.1f}s)")
                return audio
            except Exception:
                continue

    print("  No real test audio found. Using synthetic sine wave.")
    print("  For better testing, place a 16kHz WAV at scripts/test_audio/sample_16k.wav")
    return generate_test_audio()


def main() -> int:
    results = []

    # 1. Check NeMo import
    try:
        import nemo.collections.asr as nemo_asr
        results.append(check("NeMo ASR import", True))
    except ImportError as e:
        results.append(check("NeMo ASR import", False, str(e)))
        print("Install: pip install Cython packaging && pip install nemo_toolkit[asr]")
        return 1

    # 2. Check torch wasn't downgraded by NeMo
    import torch
    version = torch.__version__
    major, minor = int(version.split(".")[0]), int(version.split(".")[1])
    results.append(check(
        "PyTorch not downgraded by NeMo",
        major >= 2 and minor >= 7,
        f"torch=={version}",
    ))
    if not (major >= 2 and minor >= 7):
        print("*** NeMo downgraded PyTorch! Reinstall torch 2.7+ ***")
        print("  pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128")
        return 1

    # 3. Load model
    print("\nLoading Parakeet TDT 0.6B v2 (this takes ~30s first time)...")
    t0 = time.time()
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
        )
        model.eval()
        load_time = time.time() - t0
        results.append(check("Model loaded", True, f"{load_time:.1f}s"))
    except Exception as e:
        results.append(check("Model loaded", False, str(e)))
        return 1

    # 4. Check it's on GPU
    device = next(model.parameters()).device
    results.append(check("Model on GPU", "cuda" in str(device), str(device)))

    # 5. VRAM usage
    vram_mb = torch.cuda.memory_allocated() / (1024**2)
    results.append(check("VRAM usage", True, f"{vram_mb:.0f} MB"))

    # 6. Transcribe test audio
    print("\nTranscribing test audio...")
    audio = load_test_audio()
    t0 = time.time()
    try:
        result = model.transcribe([audio], return_hypotheses=True)
        latency = (time.time() - t0) * 1000

        # Parse result
        if isinstance(result, tuple):
            texts = result[0]
        elif isinstance(result, list):
            texts = result
        else:
            texts = [str(result)]

        text = str(texts[0]) if texts else ""
        results.append(check(
            "Transcription",
            True,
            f"'{text[:80]}' ({latency:.0f}ms)",
        ))
    except Exception as e:
        results.append(check("Transcription", False, str(e)))

    # 7. Confirm CUDA graphs disabled
    cg_env = os.environ.get("NEMO_DISABLE_CUDA_GRAPHS", "")
    results.append(check(
        "CUDA graphs disabled",
        cg_env == "1",
        f"NEMO_DISABLE_CUDA_GRAPHS={cg_env}",
    ))

    # Summary
    passed = sum(results)
    total_tests = len(results)
    print(f"\n{'=' * 40}")
    print(f"Parakeet ASR: {passed}/{total_tests} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
