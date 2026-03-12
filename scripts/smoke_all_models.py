#!/usr/bin/env python3
"""Smoke test 7: Load ALL 5 models simultaneously, measure total VRAM.

Run in lab:
    python scripts/smoke_all_models.py

This is the critical budget test. RTX 5070 has 12 GB VRAM.
Expected budget: ~5.6-6.6 GB allocated.

Models loaded in order:
1. Silero VAD (CPU, ~0 MB VRAM)
2. Dolphin AV-TSE (~2-3 GB)
3. Parakeet TDT 0.6B v2 (~2 GB)
4. Kokoro-82M TTS (~1 GB)
5. L2CS-Net ResNet50 (~0.3 GB)
"""

import os
import sys
import time

os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def vram_mb() -> float:
    import torch
    return torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0


def peak_vram_mb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("*** CUDA not available ***")
        return 1

    torch.cuda.reset_peak_memory_stats()
    results = []
    vram_log = []

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_vram:.0f} MB")
    print(f"{'=' * 60}\n")

    baseline = vram_mb()
    vram_log.append(("baseline", baseline))

    # 1. Silero VAD (CPU)
    print("1/5 Loading Silero VAD (CPU)...")
    t0 = time.time()
    try:
        vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", trust_repo=True)
        vad_model.eval()
        v = vram_mb()
        vram_log.append(("silero_vad", v))
        results.append(check("Silero VAD", True, f"loaded in {time.time()-t0:.1f}s, VRAM: {v:.0f} MB (+{v-baseline:.0f})"))
    except Exception as e:
        results.append(check("Silero VAD", False, str(e)))

    # 2. Dolphin AV-TSE (GPU)
    print("\n2/5 Loading Dolphin AV-TSE (GPU)...")
    prev = vram_mb()
    t0 = time.time()
    try:
        from look2hear.models import Dolphin
        dolphin = Dolphin.from_pretrained("JusperLee/Dolphin")
        dolphin = dolphin.to("cuda")
        dolphin.eval()
        v = vram_mb()
        delta = v - prev
        vram_log.append(("dolphin", v))
        results.append(check("Dolphin", True, f"loaded in {time.time()-t0:.1f}s, VRAM: {v:.0f} MB (+{delta:.0f})"))
    except Exception as e:
        results.append(check("Dolphin", False, str(e)))

    # 3. Parakeet TDT (GPU)
    print("\n3/5 Loading Parakeet TDT 0.6B v2 (GPU)...")
    prev = vram_mb()
    t0 = time.time()
    try:
        import nemo.collections.asr as nemo_asr
        parakeet = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        parakeet.eval()
        v = vram_mb()
        delta = v - prev
        vram_log.append(("parakeet", v))
        results.append(check("Parakeet TDT", True, f"loaded in {time.time()-t0:.1f}s, VRAM: {v:.0f} MB (+{delta:.0f})"))
    except Exception as e:
        results.append(check("Parakeet TDT", False, str(e)))

    # 4. Kokoro TTS (GPU/CPU)
    print("\n4/5 Loading Kokoro-82M TTS...")
    prev = vram_mb()
    t0 = time.time()
    try:
        from kokoro import KPipeline
        kokoro = KPipeline(lang_code="a")
        v = vram_mb()
        delta = v - prev
        vram_log.append(("kokoro", v))
        results.append(check("Kokoro TTS", True, f"loaded in {time.time()-t0:.1f}s, VRAM: {v:.0f} MB (+{delta:.0f})"))
    except Exception as e:
        results.append(check("Kokoro TTS", False, str(e)))

    # 5. L2CS-Net (GPU)
    print("\n5/5 Loading L2CS-Net (GPU)...")
    prev = vram_mb()
    t0 = time.time()
    try:
        from l2cs import Pipeline as L2CSPipeline
        l2cs = L2CSPipeline(weights=None, arch="ResNet50", device=torch.device("cuda"))
        v = vram_mb()
        delta = v - prev
        vram_log.append(("l2cs", v))
        results.append(check("L2CS-Net", True, f"loaded in {time.time()-t0:.1f}s, VRAM: {v:.0f} MB (+{delta:.0f})"))
    except Exception as e:
        results.append(check("L2CS-Net", False, str(e)))

    # Summary
    final_vram = vram_mb()
    final_peak = peak_vram_mb()

    print(f"\n{'=' * 60}")
    print("VRAM BUDGET REPORT")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Cumulative MB':>15}")
    print(f"{'-'*20} {'-'*15}")
    for name, v in vram_log:
        print(f"{name:<20} {v:>12.0f} MB")
    print(f"{'-'*20} {'-'*15}")
    print(f"{'TOTAL ALLOCATED':<20} {final_vram:>12.0f} MB")
    print(f"{'PEAK ALLOCATED':<20} {final_peak:>12.0f} MB")
    print(f"{'GPU TOTAL':<20} {total_vram:>12.0f} MB")
    print(f"{'HEADROOM':<20} {total_vram - final_peak:>12.0f} MB")
    print(f"{'=' * 60}")

    budget_ok = final_peak < total_vram * 0.85  # 85% threshold
    results.append(check(
        "VRAM budget",
        budget_ok,
        f"{final_peak:.0f}/{total_vram:.0f} MB ({final_peak/total_vram*100:.0f}%)",
    ))

    passed = sum(results)
    total_tests = len(results)
    print(f"\nAll models: {passed}/{total_tests} passed")

    if budget_ok:
        print("VRAM budget OK — all models fit in GPU memory with headroom for inference.")
    else:
        print("*** VRAM budget TIGHT — consider FP16 for Kokoro or moving Silero to CPU ***")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
