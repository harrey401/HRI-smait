#!/usr/bin/env bash
# ============================================================
# SMAIT v3 Lab Day Runbook
# ============================================================
# Master script: runs all smoke tests in order, stops on first failure.
#
# Usage:
#   bash scripts/lab_runbook.sh          # Full runbook
#   bash scripts/lab_runbook.sh --skip-setup  # Skip env setup
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SKIP_SETUP=false
if [[ "${1:-}" == "--skip-setup" ]]; then
    SKIP_SETUP=true
fi

phase() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
}

run_test() {
    local name="$1"
    local script="$2"
    echo -e "\n${YELLOW}--- $name ---${NC}"
    if python3 "$script"; then
        echo -e "${GREEN}[PASSED] $name${NC}"
        return 0
    else
        echo -e "${RED}[FAILED] $name${NC}"
        echo -e "${RED}Fix the issue above before continuing.${NC}"
        return 1
    fi
}

START_TIME=$(date +%s)

# ---- Phase 0: Environment Setup ----
if [ "$SKIP_SETUP" = false ]; then
    phase "Phase 0: Environment Setup"
    bash scripts/setup_lab.sh
fi

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# ---- Phase 1: Generate Test Audio ----
phase "Phase 1: Generate Test Audio Fixtures"
python3 scripts/generate_test_audio.py

# ---- Phase 2: Individual Smoke Tests ----
phase "Phase 2: Individual Model Smoke Tests"

run_test "GPU/CUDA Validation" scripts/smoke_torch.py
run_test "Silero VAD" scripts/smoke_vad.py
run_test "Parakeet ASR" scripts/smoke_parakeet.py
run_test "Dolphin AV-TSE" scripts/smoke_dolphin.py
run_test "Kokoro TTS" scripts/smoke_tts.py
run_test "L2CS-Net Gaze" scripts/smoke_gaze.py

# ---- Phase 3: Combined VRAM Budget ----
phase "Phase 3: All Models Combined (VRAM Budget)"
run_test "All Models VRAM Budget" scripts/smoke_all_models.py

# ---- Phase 4: E2E Pipeline Test ----
phase "Phase 4: End-to-End Pipeline Test"
run_test "E2E Pipeline" scripts/test_e2e_pipeline.py

# ---- Phase 5: Run Unit Tests ----
phase "Phase 5: Unit Test Suite"
echo "Running pytest..."
python3 -m pytest tests/ -v --tb=short || {
    echo -e "${YELLOW}Some unit tests failed. Review above.${NC}"
}

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

phase "Lab Day Complete"
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Next steps:"
echo "  1. Connect Jackie to ws://<your-ip>:8765"
echo "  2. python run_jackie.py --debug"
echo "  3. Test full conversation loop"
echo "  4. Measure latency (target: speech-end to TTS-start < 1500ms)"
echo ""
echo "If all smoke tests passed, the system is ready for Jackie integration."
