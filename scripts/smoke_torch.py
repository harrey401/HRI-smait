#!/usr/bin/env python3
"""Smoke test 1: Verify PyTorch + CUDA + Blackwell (sm_120) on RTX 5070.

Run in lab:
    python scripts/smoke_torch.py

Expected output:
    [PASS] PyTorch version >= 2.7.0
    [PASS] CUDA available
    [PASS] GPU: NVIDIA GeForce RTX 5070 (or similar)
    [PASS] CUDA compute capability >= 12.0 (sm_120)
    [PASS] Tensor ops on GPU
    [PASS] VRAM: X.X GB total, X.X GB free
"""

import sys


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main() -> int:
    results = []

    # 1. PyTorch version
    import torch

    version = torch.__version__
    major, minor = int(version.split(".")[0]), int(version.split(".")[1])
    results.append(check(
        "PyTorch version",
        major >= 2 and minor >= 7,
        f"{version} (need >= 2.7.0 for sm_120)",
    ))

    # 2. CUDA available
    cuda_ok = torch.cuda.is_available()
    results.append(check("CUDA available", cuda_ok))
    if not cuda_ok:
        print("\n*** CUDA not available. Cannot continue GPU tests. ***")
        return 1

    # 3. GPU name
    gpu_name = torch.cuda.get_device_name(0)
    results.append(check("GPU detected", True, gpu_name))

    # 4. Compute capability (sm_120 = 12.0)
    cap = torch.cuda.get_device_capability(0)
    cap_str = f"{cap[0]}.{cap[1]}"
    results.append(check(
        "Compute capability",
        cap[0] >= 12,
        f"sm_{cap[0]}{cap[1]}0 ({cap_str}) — need >= 12.0 for Blackwell",
    ))

    # 5. Tensor operations
    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        c = a @ b
        assert c.shape == (256, 256)
        # Also test float16 (important for model inference)
        a16 = a.half()
        b16 = b.half()
        c16 = a16 @ b16
        assert c16.shape == (256, 256)
        results.append(check("Tensor ops on GPU", True, "float32 + float16 matmul OK"))
    except Exception as e:
        results.append(check("Tensor ops on GPU", False, str(e)))

    # 6. VRAM
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = (total - torch.cuda.memory_allocated(0) / (1024**3))
    results.append(check(
        "VRAM",
        total >= 6,
        f"{total:.1f} GB total, ~{free:.1f} GB free",
    ))

    # 7. torchaudio
    try:
        import torchaudio
        results.append(check("torchaudio", True, torchaudio.__version__))
    except ImportError:
        results.append(check("torchaudio", False, "not installed"))

    # Summary
    passed = sum(results)
    total_tests = len(results)
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total_tests} passed")

    if all(results):
        print("GPU is ready for SMAIT models.")
        return 0
    else:
        print("*** Fix failures above before proceeding. ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
