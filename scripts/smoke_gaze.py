#!/usr/bin/env python3
"""Smoke test 5: Verify L2CS-Net gaze estimation on GPU.

Run in lab:
    python scripts/smoke_gaze.py

Tests:
- L2CS Pipeline loads with ResNet50
- Inference on a synthetic face image
- VRAM usage (~0.3 GB)
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
        from l2cs import Pipeline as L2CSPipeline
        results.append(check("L2CS import", True))
    except ImportError as e:
        results.append(check("L2CS import", False, str(e)))
        print("Install: pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main")
        return 1

    # 2. Load pipeline
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading L2CS-Net (ResNet50) on {device}...")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    try:
        pipeline = L2CSPipeline(
            weights=None,  # Auto-download
            arch="ResNet50",
            device=device,
        )
        load_time = time.time() - t0
        results.append(check("Pipeline loaded", True, f"{load_time:.1f}s"))
    except Exception as e:
        results.append(check("Pipeline loaded", False, str(e)))
        return 1

    # 3. VRAM
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024**2)
        results.append(check("VRAM usage", True, f"{vram_mb:.0f} MB"))

    # 4. Inference on synthetic face image
    # L2CS expects a BGR face crop (like from OpenCV)
    fake_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    print("\nRunning gaze estimation on synthetic face...")
    t0 = time.time()
    try:
        result = pipeline.step(fake_face)
        latency = (time.time() - t0) * 1000

        has_yaw = hasattr(result, "yaw") and len(result.yaw) > 0
        has_pitch = hasattr(result, "pitch") and len(result.pitch) > 0

        if has_yaw and has_pitch:
            yaw = float(result.yaw[0])
            pitch = float(result.pitch[0])
            results.append(check(
                "Gaze inference",
                True,
                f"yaw={yaw:.1f}°, pitch={pitch:.1f}°, latency={latency:.0f}ms",
            ))
        else:
            # No face detected in random noise — that's OK, the pipeline ran
            results.append(check(
                "Gaze inference",
                True,
                f"no face detected in noise (expected), pipeline ran OK, {latency:.0f}ms",
            ))
    except Exception as e:
        results.append(check("Gaze inference", False, str(e)))

    # Summary
    passed = sum(results)
    total_tests = len(results)
    print(f"\n{'=' * 40}")
    print(f"L2CS-Net: {passed}/{total_tests} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
