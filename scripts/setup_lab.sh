#!/usr/bin/env bash
# ============================================================
# SMAIT v3 Lab Environment Setup
# ============================================================
# Run this FIRST when you arrive at the lab.
# It installs all dependencies in the correct order and
# verifies nothing gets downgraded.
#
# Usage:
#   cd ~/projects/SMAIT-v3   (or wherever the repo is in lab)
#   bash scripts/setup_lab.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
step() { echo -e "\n${YELLOW}=== $1 ===${NC}"; }

# ---- Pre-flight checks ----
step "Pre-flight checks"

# Check NVIDIA driver
if ! nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Install NVIDIA driver R570+."
fi
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
pass "NVIDIA driver: $DRIVER"

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
pass "GPU: $GPU_NAME"

# Check CUDA
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
pass "Driver version: $CUDA_VER"

# Check Python
PYTHON_VER=$(python3 --version 2>&1)
pass "Python: $PYTHON_VER"

# Check venv
if [ -d "venv" ]; then
    source venv/bin/activate
    pass "Virtual env activated"
else
    warn "No venv found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pass "Virtual env created and activated"
fi

# ---- Step 1: PyTorch + CUDA ----
step "Step 1: PyTorch 2.7 + CUDA 12.8"

pip install --upgrade pip

# Install PyTorch with CUDA 12.8
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# VERIFY torch version
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
if [[ "$TORCH_VER" == 2.7* ]]; then
    pass "torch==$TORCH_VER"
else
    fail "torch==$TORCH_VER (expected 2.7.x). Something went wrong."
fi

# VERIFY CUDA
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_OK" = "True" ]; then
    pass "CUDA available"
else
    fail "CUDA not available in PyTorch"
fi

# ---- Step 2: NeMo (Parakeet ASR) ----
step "Step 2: NeMo toolkit (Parakeet ASR)"

pip install Cython packaging
pip install nemo_toolkit[asr]

# CRITICAL: Verify NeMo didn't downgrade torch
TORCH_AFTER=$(python3 -c "import torch; print(torch.__version__)")
if [[ "$TORCH_AFTER" == 2.7* ]]; then
    pass "torch still $TORCH_AFTER after NeMo install"
else
    fail "NeMo DOWNGRADED torch to $TORCH_AFTER! Reinstalling torch..."
    pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    TORCH_AFTER2=$(python3 -c "import torch; print(torch.__version__)")
    if [[ "$TORCH_AFTER2" == 2.7* ]]; then
        warn "Reinstalled torch==$TORCH_AFTER2. NeMo may have compatibility issues."
    else
        fail "Could not restore torch 2.7. Manual intervention needed."
    fi
fi

# ---- Step 3: Kokoro TTS ----
step "Step 3: Kokoro TTS"

pip install "kokoro>=0.9.4" soundfile

# espeak-ng is needed for phonemization
if ! command -v espeak-ng &>/dev/null; then
    warn "espeak-ng not found. Installing..."
    sudo apt-get install -y espeak-ng || warn "Could not install espeak-ng. Kokoro may have issues."
fi

python3 -c "from kokoro import KPipeline; print('OK')" && pass "Kokoro importable" || fail "Kokoro import failed"

# ---- Step 4: L2CS-Net ----
step "Step 4: L2CS-Net (gaze estimation)"

pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main

python3 -c "from l2cs import Pipeline; print('OK')" && pass "L2CS importable" || fail "L2CS import failed"

# ---- Step 5: Other dependencies ----
step "Step 5: Remaining dependencies"

pip install -r requirements.txt

# FINAL: Verify torch wasn't touched
TORCH_FINAL=$(python3 -c "import torch; print(torch.__version__)")
if [[ "$TORCH_FINAL" == 2.7* ]]; then
    pass "torch still $TORCH_FINAL after all installs"
else
    fail "torch was downgraded to $TORCH_FINAL during pip install -r requirements.txt!"
fi

# ---- Step 6: JDK check (for Android builds) ----
step "Step 6: JDK check"

if command -v javac &>/dev/null; then
    JAVA_VER=$(javac -version 2>&1)
    pass "JDK: $JAVA_VER"
else
    warn "JDK not found. Install for Android builds: sudo apt install openjdk-17-jdk"
fi

# ---- Summary ----
step "Setup Complete"

echo ""
echo "All dependencies installed. Next steps:"
echo "  1. python scripts/smoke_torch.py        # GPU validation"
echo "  2. python scripts/smoke_vad.py           # Silero VAD"
echo "  3. python scripts/smoke_parakeet.py      # Parakeet ASR"
echo "  4. python scripts/smoke_dolphin.py       # Dolphin separation"
echo "  5. python scripts/smoke_tts.py           # Kokoro TTS"
echo "  6. python scripts/smoke_gaze.py          # L2CS-Net gaze"
echo "  7. python scripts/smoke_all_models.py    # All models + VRAM budget"
echo ""
echo "Or run them all at once:"
echo "  bash scripts/lab_runbook.sh"
