#!/usr/bin/env python3
"""Smoke test 3: Verify Dolphin AV-TSE model on GPU.

Run in lab:
    python scripts/smoke_dolphin.py

Tests:
- Model loads from pretrained
- Audio tensor shape [1, samples]
- Video tensor shape [1, 1, T, 88, 88, 1]
- Forward pass produces output
- VRAM usage
"""

import sys
import time

import numpy as np
import torch


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main() -> int:
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("*** CUDA not available ***")
        return 1

    # 1. Import Dolphin
    try:
        from look2hear.models import Dolphin
        results.append(check("Dolphin import", True))
    except ImportError as e:
        results.append(check("Dolphin import", False, str(e)))
        print("Vendor look2hear/ from github.com/JusperLee/Dolphin into project root")
        return 1

    # 2. Load model
    print("\nLoading Dolphin model...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model = Dolphin.from_pretrained("JusperLee/Dolphin")
        model = model.to(device)
        model.eval()
        load_time = time.time() - t0
        results.append(check("Model loaded", True, f"{load_time:.1f}s"))
    except Exception as e:
        results.append(check("Model loaded", False, str(e)))
        return 1

    # 3. VRAM after load
    vram_mb = torch.cuda.memory_allocated() / (1024**2)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    results.append(check("VRAM usage", True, f"{vram_mb:.0f} MB allocated, {peak_mb:.0f} MB peak"))

    # 4. Audio tensor shape test
    # Simulate 2 seconds of 16kHz mono audio
    n_samples = 16000 * 2
    audio = torch.randn(1, n_samples, device=device)
    results.append(check("Audio tensor shape", audio.shape == (1, n_samples), str(audio.shape)))

    # 5. Video tensor shape test
    # Simulate 25 frames of 88x88 grayscale lip ROI
    n_frames = 25
    video = torch.randn(1, 1, n_frames, 88, 88, 1, device=device)
    results.append(check("Video tensor shape", video.shape == (1, 1, n_frames, 88, 88, 1), str(video.shape)))

    # 6. Forward pass
    print("\nRunning Dolphin inference...")
    t0 = time.time()
    try:
        with torch.inference_mode():
            output = model(audio, video)

        latency = (time.time() - t0) * 1000

        if isinstance(output, tuple):
            separated = output[0]
            detail = f"tuple output, shape={separated.shape}, latency={latency:.0f}ms"
        else:
            separated = output
            detail = f"tensor output, shape={separated.shape}, latency={latency:.0f}ms"

        results.append(check("Forward pass", True, detail))
    except Exception as e:
        results.append(check("Forward pass", False, str(e)))
        print("\n*** This is the highest-risk test. Possible issues:")
        print("  - Video tensor shape mismatch (try different dim ordering)")
        print("  - VRAM OOM (reduce n_frames or n_samples)")
        print("  - sm_120 kernel issues")

    # 7. Peak VRAM during inference
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    results.append(check("Peak VRAM (with inference)", True, f"{peak_mb:.0f} MB"))

    # Summary
    passed = sum(results)
    total_tests = len(results)
    print(f"\n{'=' * 40}")
    print(f"Dolphin: {passed}/{total_tests} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
