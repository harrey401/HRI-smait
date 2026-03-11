#!/usr/bin/env python3
"""Pre-download all model weights before lab day.

Run at HOME (needs internet, no GPU required):
    python scripts/download_weights.py

Downloads:
1. Dolphin AV-TSE (via HuggingFace Hub)
2. Parakeet TDT 0.6B v2 (via NeMo/HuggingFace)
3. Kokoro-82M (via pip, weights bundled)
4. L2CS-Net ResNet50 (via Google Drive / auto-download)
5. Silero VAD (via torch.hub)

Run this script BEFORE going to the lab so you're not waiting
on downloads over lab WiFi.
"""

import os
import sys
import time


def step(name: str) -> None:
    print(f"\n{'='*50}")
    print(f"  Downloading: {name}")
    print(f"{'='*50}")


def main() -> int:
    errors = []

    # 1. Silero VAD (torch.hub cache)
    step("Silero VAD")
    try:
        import torch
        model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", trust_repo=True)
        print("  OK: Silero VAD cached")
    except Exception as e:
        print(f"  FAILED: {e}")
        errors.append("Silero VAD")

    # 2. Dolphin AV-TSE
    step("Dolphin AV-TSE")
    try:
        # Check if look2hear is vendored
        from look2hear.models import Dolphin
        # This downloads the pretrained weights to HF cache
        model = Dolphin.from_pretrained("JusperLee/Dolphin")
        del model
        print("  OK: Dolphin weights cached")
    except ImportError:
        print("  SKIP: look2hear not vendored yet. Vendor it from github.com/JusperLee/Dolphin")
        print("  The weights will be downloaded on first load in lab.")
    except Exception as e:
        print(f"  FAILED: {e}")
        errors.append("Dolphin")

    # 3. Parakeet TDT
    step("Parakeet TDT 0.6B v2")
    try:
        os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"
        import nemo.collections.asr as nemo_asr
        # This downloads the model. On CPU it won't load to GPU but caches the weights.
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
        )
        del model
        print("  OK: Parakeet weights cached")
    except ImportError:
        print("  SKIP: NeMo not installed. Install in lab: pip install nemo_toolkit[asr]")
    except Exception as e:
        # May fail on CPU — that's OK, weights are still cached
        print(f"  Note: {e}")
        print("  Weights may still be cached. Will verify in lab.")

    # 4. Kokoro-82M
    step("Kokoro-82M TTS")
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")
        # Quick test to ensure weights are loaded
        for _, _, audio in pipeline("test", voice="af_heart", speed=1.0):
            break
        del pipeline
        print("  OK: Kokoro weights cached")
    except ImportError:
        print("  SKIP: Kokoro not installed. Install: pip install kokoro>=0.9.4")
    except Exception as e:
        print(f"  FAILED: {e}")
        errors.append("Kokoro")

    # 5. L2CS-Net
    step("L2CS-Net ResNet50")
    try:
        from l2cs import Pipeline as L2CSPipeline
        import torch
        # This triggers weight download
        pipeline = L2CSPipeline(weights=None, arch="ResNet50", device=torch.device("cpu"))
        del pipeline
        print("  OK: L2CS-Net weights cached")
    except ImportError:
        print("  SKIP: L2CS not installed. Install: pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main")
    except Exception as e:
        print(f"  Note: {e}")
        print("  L2CS weights are hosted on Google Drive. If download failed,")
        print("  try again or manually download to ~/.l2cs/ cache dir.")

    # Summary
    print(f"\n{'='*50}")
    if errors:
        print(f"FAILURES: {', '.join(errors)}")
        print("Fix these before lab day.")
        return 1
    else:
        print("All available weights downloaded and cached.")
        print("You're ready for lab day.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
