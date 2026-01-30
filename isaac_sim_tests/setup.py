"""
SMAIT HRI v2.0 - Isaac Sim Test Setup

Run this script to set up the testing environment.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check for required Python packages"""
    required = ['numpy', 'pyyaml', 'scipy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            *missing, '--break-system-packages'
        ])
        print("Done!")
    else:
        print("All dependencies installed!")


def create_directories():
    """Create required directories"""
    dirs = [
        'audio_samples/short_phrases',
        'audio_samples/long_speech',
        'audio_samples/noise',
        'results',
        'environments'
    ]
    
    base = Path(__file__).parent
    
    for d in dirs:
        path = base / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")


def generate_test_audio():
    """Generate synthetic test audio files"""
    try:
        from domain_randomization.noise_mixer import generate_test_noises
        generate_test_noises("./audio_samples/noise")
    except Exception as e:
        print(f"Could not generate test noises: {e}")
        print("You can create them manually or download real recordings")


def print_instructions():
    """Print setup instructions"""
    print("""
================================================================================
SMAIT HRI v2.0 - Isaac Sim Test Setup Complete
================================================================================

NEXT STEPS:

1. INSTALL AUDIO2FACE
   - Open NVIDIA Omniverse Launcher
   - Go to "Exchange" tab
   - Search "Audio2Face"
   - Click Install
   
2. PREPARE TEST AUDIO
   - Add speech WAV files to: audio_samples/short_phrases/
   - Add noise WAV files to: audio_samples/noise/
   - Format: 16kHz mono WAV recommended
   
3. LAUNCH AUDIO2FACE
   - Start Audio2Face from Omniverse Launcher
   - Load a character (File > Load Character)
   - Enable streaming (Window > Streaming > Enable)
   
4. RUN TESTS
   
   Quick test (mock mode, no Isaac Sim):
   $ python run_test.py --quick
   
   Phase 1 - ASD Parameter Tuning:
   $ python run_test.py --phase 1
   
   All phases:
   $ python run_test.py --all
   
   From Isaac Sim Python (for real simulation):
   $ ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh run_test.py --all

5. VIEW RESULTS
   Results are saved in: ./results/<timestamp>/
   - metrics.csv: Raw data
   - summary.json: Statistics
   - report.txt: Human-readable report

================================================================================
""")


def main():
    print("Setting up SMAIT HRI v2.0 Isaac Sim Tests...")
    print()
    
    check_dependencies()
    create_directories()
    generate_test_audio()
    print_instructions()


if __name__ == "__main__":
    main()
