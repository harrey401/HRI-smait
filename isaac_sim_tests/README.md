# SMAIT HRI v2.0 - Isaac Sim Domain Randomization Testing
# Configuration and Test Plan

## Overview

This test suite validates the SMAIT HRI system parameters through
simulated scenarios with controlled ground truth.

## Prerequisites

### 1. Isaac Sim (Already Installed)
- Version: 4.5.0 ✓

### 2. Audio2Face (Need to Install)
Download from NVIDIA Omniverse Launcher:
1. Open Omniverse Launcher
2. Go to "Exchange" tab
3. Search "Audio2Face"
4. Install latest version (2023.2.0 or newer)

### 3. Test Audio Samples
We need clean speech audio files to drive Audio2Face:
- Short phrases (2-5 seconds)
- Long monologues (10-30 seconds) 
- Various speakers (if available)

## Test Phases

### Phase 1: ASD Parameter Tuning (Single Speaker)
Goal: Find optimal thresholds for mouth movement detection
- Single face, controlled speech
- No noise, no delay
- Sweep: min_lip_movement, min_mar_for_speech, asd_threshold

### Phase 2: Audio-Visual Delay Tolerance
Goal: Determine acceptable AV desync range
- Inject delays: -200ms to +200ms
- Measure accuracy degradation
- Find breaking point

### Phase 3: Noise Robustness
Goal: Test with background audio
- Cafe ambience
- Street noise
- Music
- Overlapping speech

### Phase 4: Articulation Variation
Goal: Handle different speaking styles
- Normal articulation
- Lazy/mumbling (damped blendshapes)
- Exaggerated (amplified blendshapes)

### Phase 5: Multi-Speaker
Goal: Correct speaker identification
- 2-3 faces in frame
- Only one speaking at a time
- Test face tracking handoff

## File Structure

```
isaac_sim_tests/
├── README.md                 # This file
├── config.yaml               # Test parameters
├── environments/
│   └── test_scene.usda       # Basic scene with Audio2Face character
├── domain_randomization/
│   ├── __init__.py
│   ├── av_delay.py           # Audio-visual delay injection
│   ├── noise_mixer.py        # Background noise injection
│   └── articulation.py       # Mouth movement scaling
├── parameter_sweep/
│   ├── sweep_runner.py       # Automated parameter testing
│   └── sweep_configs/
│       ├── asd_thresholds.yaml
│       ├── timing_params.yaml
│       └── noise_levels.yaml
├── metrics/
│   ├── collector.py          # Real-time metric collection
│   ├── analyzer.py           # Post-test analysis
│   └── visualizer.py         # Generate plots
├── audio_samples/            # Test audio files
│   ├── short_phrases/
│   ├── long_speech/
│   └── noise/
└── results/                  # Test outputs
    └── {timestamp}/
        ├── metrics.csv
        ├── config_used.yaml
        └── summary.md
```

## Quick Start

1. Install Audio2Face via Omniverse Launcher
2. Download test audio samples (or use your own)
3. Run: `python run_test.py --phase 1`

## Connection Architecture

```
┌─────────────────┐     Audio      ┌─────────────────┐
│  Test Runner    │ ─────────────► │  Audio2Face     │
│  (Python)       │                │  (Generates     │
│                 │ ◄───────────── │   face motion)  │
│                 │   Blendshapes  │                 │
└────────┬────────┘                └─────────────────┘
         │
         │ Video frames + Audio
         ▼
┌─────────────────┐
│  SMAIT HRI      │
│  System         │
│  (Under Test)   │
└────────┬────────┘
         │
         │ ASD Results
         ▼
┌─────────────────┐
│  Metrics        │
│  Collector      │
│  (Compare to    │
│   ground truth) │
└─────────────────┘
```
