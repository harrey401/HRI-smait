"""
SMAIT HRI v2.0 - Domain Randomization Modules

These modules inject various forms of variation to test system robustness:
- AV delay: Temporal misalignment between audio and video
- Noise: Background audio interference
- Articulation: Different speaking styles (lazy to exaggerated)
- Lighting: Illumination changes affecting face detection
"""

from .av_delay import AVDelayInjector
from .noise_mixer import NoiseMixer
from .articulation import ArticulationController

__all__ = [
    'AVDelayInjector',
    'NoiseMixer', 
    'ArticulationController'
]
