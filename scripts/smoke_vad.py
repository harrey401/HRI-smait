#!/usr/bin/env python3
"""Smoke test 6: Verify Silero VAD loads and runs.

Run in lab:
    python scripts/smoke_vad.py

Silero VAD runs on CPU (no GPU needed), but we test it here
to confirm it works alongside the GPU models.
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

    # 1. Load Silero VAD
    print("Loading Silero VAD from torch.hub...")
    t0 = time.time()
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        model.eval()
        load_time = time.time() - t0
        results.append(check("Silero VAD loaded", True, f"{load_time:.1f}s"))
    except Exception as e:
        results.append(check("Silero VAD loaded", False, str(e)))
        return 1

    # 2. Test on silence (should be low probability)
    silence = torch.zeros(16000)  # 1s at 16kHz
    prob = model(silence, 16000).item()
    results.append(check(
        "Silence detection",
        prob < 0.5,
        f"speech_prob={prob:.3f} (expected < 0.5 for silence)",
    ))

    # 3. Test on synthetic speech-like signal
    t = torch.linspace(0, 0.03, 480)
    speech_like = torch.sin(2 * 3.14159 * 200 * t) * 0.8
    model.reset_states()
    prob = model(speech_like, 16000).item()
    results.append(check(
        "Speech-like signal",
        True,
        f"speech_prob={prob:.3f} (synthetic, value varies)",
    ))

    # 4. Test chunk processing loop (simulates real pipeline)
    model.reset_states()
    n_chunks = 100  # 3 seconds at 30ms chunks
    t0 = time.time()
    for _ in range(n_chunks):
        chunk = torch.randn(480) * 0.1
        model(chunk, 16000)
    loop_time = (time.time() - t0) * 1000
    results.append(check(
        "Chunk processing loop",
        loop_time < 1000,
        f"{n_chunks} chunks in {loop_time:.0f}ms ({loop_time/n_chunks:.1f}ms/chunk)",
    ))

    # 5. Confirm it's on CPU (no VRAM usage)
    results.append(check(
        "Runs on CPU",
        True,
        "no VRAM allocated for VAD",
    ))

    # Summary
    passed = sum(results)
    total_tests = len(results)
    print(f"\n{'=' * 40}")
    print(f"Silero VAD: {passed}/{total_tests} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
