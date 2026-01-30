"""
SMAIT HRI v2.0 - Articulation Controller

Controls the intensity of mouth movements to test ASD robustness
across different speaking styles.

Articulation styles:
- Mumbling (0.3-0.5x): Minimal mouth movement
- Lazy (0.6-0.8x): Reduced articulation  
- Normal (1.0x): Standard Audio2Face output
- Exaggerated (1.3-1.5x): Over-articulated speech
- Very exaggerated (1.6-2.0x): Theatrical speech

This affects the VISUAL signal only, not the audio.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ArticulationStyle(Enum):
    """Predefined articulation styles"""
    MUMBLING = 0.4
    LAZY = 0.6
    NORMAL = 1.0
    CLEAR = 1.2
    EXAGGERATED = 1.5
    THEATRICAL = 2.0


@dataclass
class BlendshapeWeights:
    """Weights for different blendshape groups"""
    jaw: float = 1.0          # jawOpen, jawForward, etc.
    lips: float = 1.0         # All lip-related blendshapes
    mouth_corners: float = 1.0 # Smile, frown, etc.
    cheeks: float = 1.0       # Puff, squint
    tongue: float = 1.0       # If available


class ArticulationController:
    """
    Controls mouth movement intensity for testing ASD across
    different articulation styles.
    
    The controller scales blendshape values from Audio2Face
    to simulate different speaking styles.
    """
    
    # Blendshape categories (ARKit naming)
    JAW_BLENDSHAPES = {
        'jawOpen', 'jawForward', 'jawLeft', 'jawRight'
    }
    
    LIP_BLENDSHAPES = {
        'mouthClose', 'mouthFunnel', 'mouthPucker',
        'mouthRollLower', 'mouthRollUpper',
        'mouthShrugLower', 'mouthShrugUpper',
        'mouthLowerDownLeft', 'mouthLowerDownRight',
        'mouthUpperUpLeft', 'mouthUpperUpRight',
        'mouthPressLeft', 'mouthPressRight'
    }
    
    CORNER_BLENDSHAPES = {
        'mouthLeft', 'mouthRight',
        'mouthSmileLeft', 'mouthSmileRight',
        'mouthFrownLeft', 'mouthFrownRight',
        'mouthDimpleLeft', 'mouthDimpleRight',
        'mouthStretchLeft', 'mouthStretchRight'
    }
    
    def __init__(self, initial_scale: float = 1.0):
        """
        Args:
            initial_scale: Global scaling factor (1.0 = normal)
        """
        self.global_scale = initial_scale
        self.weights = BlendshapeWeights()
        
        # Custom per-blendshape overrides
        self._overrides: Dict[str, float] = {}
        
        # Noise injection for realism
        self._add_noise = False
        self._noise_level = 0.02
    
    def set_style(self, style: ArticulationStyle):
        """Set articulation to a predefined style"""
        self.global_scale = style.value
        
        # Adjust individual weights based on style
        if style == ArticulationStyle.MUMBLING:
            self.weights = BlendshapeWeights(
                jaw=0.3,      # Very little jaw movement
                lips=0.5,     # Some lip movement
                mouth_corners=0.4,
                cheeks=0.3,
                tongue=0.2
            )
        elif style == ArticulationStyle.LAZY:
            self.weights = BlendshapeWeights(
                jaw=0.6,
                lips=0.7,
                mouth_corners=0.6,
                cheeks=0.5,
                tongue=0.5
            )
        elif style == ArticulationStyle.NORMAL:
            self.weights = BlendshapeWeights()  # All 1.0
        elif style == ArticulationStyle.CLEAR:
            self.weights = BlendshapeWeights(
                jaw=1.2,
                lips=1.3,
                mouth_corners=1.1,
                cheeks=1.0,
                tongue=1.2
            )
        elif style == ArticulationStyle.EXAGGERATED:
            self.weights = BlendshapeWeights(
                jaw=1.5,
                lips=1.6,
                mouth_corners=1.4,
                cheeks=1.3,
                tongue=1.5
            )
        elif style == ArticulationStyle.THEATRICAL:
            self.weights = BlendshapeWeights(
                jaw=2.0,
                lips=2.0,
                mouth_corners=1.8,
                cheeks=1.5,
                tongue=1.8
            )
        
        print(f"[ARTICULATION] Style set to {style.name} (scale={style.value})")
    
    def set_scale(self, scale: float):
        """Set global articulation scale"""
        self.global_scale = np.clip(scale, 0.1, 3.0)
        print(f"[ARTICULATION] Global scale: {self.global_scale:.2f}")
    
    def set_blendshape_override(self, name: str, scale: float):
        """Override scale for a specific blendshape"""
        self._overrides[name] = scale
    
    def clear_overrides(self):
        """Clear all blendshape overrides"""
        self._overrides.clear()
    
    def enable_noise(self, level: float = 0.02):
        """Enable random noise on blendshapes"""
        self._add_noise = True
        self._noise_level = level
    
    def disable_noise(self):
        """Disable random noise"""
        self._add_noise = False
    
    def process_blendshapes(
        self, 
        blendshapes: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Process blendshapes through articulation scaling.
        
        Args:
            blendshapes: Dictionary of blendshape_name -> value
            
        Returns:
            Scaled blendshape values
        """
        result = {}
        
        for name, value in blendshapes.items():
            # Get the appropriate weight
            if name in self._overrides:
                weight = self._overrides[name]
            elif name in self.JAW_BLENDSHAPES:
                weight = self.weights.jaw
            elif name in self.LIP_BLENDSHAPES:
                weight = self.weights.lips
            elif name in self.CORNER_BLENDSHAPES:
                weight = self.weights.mouth_corners
            else:
                weight = 1.0
            
            # Apply global scale and category weight
            scaled = value * self.global_scale * weight
            
            # Add noise if enabled
            if self._add_noise:
                noise = np.random.normal(0, self._noise_level)
                scaled += noise
            
            # Clamp to valid range [0, 1]
            result[name] = np.clip(scaled, 0.0, 1.0)
        
        return result
    
    def compute_mar_from_blendshapes(
        self, 
        blendshapes: Dict[str, float]
    ) -> float:
        """
        Compute Mouth Aspect Ratio from blendshapes.
        
        This provides ground truth MAR from Audio2Face,
        which can be compared to MediaPipe-detected MAR.
        """
        jaw_open = blendshapes.get('jawOpen', 0.0)
        mouth_close = blendshapes.get('mouthClose', 0.0)
        mouth_funnel = blendshapes.get('mouthFunnel', 0.0)
        mouth_pucker = blendshapes.get('mouthPucker', 0.0)
        
        # Approximate MAR from blendshapes
        # jawOpen increases vertical distance
        # mouthClose decreases it
        # mouthFunnel/Pucker affect shape but less directly
        
        vertical = jaw_open * 0.3 - mouth_close * 0.1 + mouth_funnel * 0.1
        
        # Assume horizontal stays relatively constant (mouth width)
        # This is a simplified approximation
        
        # Normalize to typical MAR range [0.0, 0.5]
        mar = np.clip(vertical, 0.0, 0.5)
        
        return mar
    
    def is_speaking(
        self, 
        blendshapes: Dict[str, float],
        threshold: float = 0.1
    ) -> bool:
        """
        Determine if the character is speaking based on blendshapes.
        
        This provides GROUND TRUTH for testing.
        """
        mar = self.compute_mar_from_blendshapes(blendshapes)
        return mar > threshold
    
    def get_ground_truth(
        self,
        blendshapes: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Get ground truth speaking state from blendshapes.
        
        Returns dictionary with:
        - is_speaking: Boolean
        - mar: Estimated MAR value
        - jaw_open: Raw jawOpen value
        - confidence: How confident we are in the label
        """
        mar = self.compute_mar_from_blendshapes(blendshapes)
        jaw_open = blendshapes.get('jawOpen', 0.0)
        
        is_speaking = mar > 0.05
        
        # Confidence is higher for clear cases (very open or very closed)
        if mar > 0.15:
            confidence = 0.95
        elif mar > 0.08:
            confidence = 0.8
        elif mar < 0.02:
            confidence = 0.95
        else:
            confidence = 0.6
        
        return {
            'is_speaking': is_speaking,
            'mar': mar,
            'jaw_open': jaw_open,
            'confidence': confidence
        }


class ArticulationSequencer:
    """
    Generates sequences of articulation changes for testing.
    
    Useful for automated testing across different styles
    within a single test run.
    """
    
    def __init__(self, controller: ArticulationController):
        self.controller = controller
        self._sequence: list = []
        self._current_index = 0
    
    def add_step(
        self, 
        style: ArticulationStyle, 
        duration_seconds: float
    ):
        """Add a step to the sequence"""
        self._sequence.append({
            'style': style,
            'duration': duration_seconds
        })
    
    def create_sweep(
        self,
        styles: list,
        duration_per_style: float = 5.0
    ):
        """Create a sweep through multiple styles"""
        self._sequence.clear()
        for style in styles:
            self.add_step(style, duration_per_style)
    
    def create_standard_sweep(self, duration_per_style: float = 5.0):
        """Create standard sweep through all styles"""
        self.create_sweep([
            ArticulationStyle.MUMBLING,
            ArticulationStyle.LAZY,
            ArticulationStyle.NORMAL,
            ArticulationStyle.CLEAR,
            ArticulationStyle.EXAGGERATED
        ], duration_per_style)
    
    def reset(self):
        """Reset sequence to beginning"""
        self._current_index = 0
    
    def get_current_style(self) -> Optional[ArticulationStyle]:
        """Get current style in sequence"""
        if self._current_index < len(self._sequence):
            return self._sequence[self._current_index]['style']
        return None
    
    def advance(self) -> bool:
        """Move to next step in sequence"""
        if self._current_index < len(self._sequence):
            self._current_index += 1
            if self._current_index < len(self._sequence):
                style = self._sequence[self._current_index]['style']
                self.controller.set_style(style)
                return True
        return False


def test_articulation_controller():
    """Test the articulation controller"""
    print("Testing ArticulationController")
    print("=" * 40)
    
    controller = ArticulationController()
    
    # Test blendshapes (simulated Audio2Face output)
    test_blendshapes = {
        'jawOpen': 0.3,
        'mouthClose': 0.0,
        'mouthFunnel': 0.1,
        'mouthSmileLeft': 0.2,
        'mouthSmileRight': 0.2
    }
    
    print("\nOriginal blendshapes:")
    for name, value in test_blendshapes.items():
        print(f"  {name}: {value:.3f}")
    
    # Test different styles
    for style in ArticulationStyle:
        controller.set_style(style)
        scaled = controller.process_blendshapes(test_blendshapes)
        ground_truth = controller.get_ground_truth(scaled)
        
        print(f"\n{style.name}:")
        print(f"  jawOpen: {test_blendshapes['jawOpen']:.3f} -> {scaled['jawOpen']:.3f}")
        print(f"  Ground truth: speaking={ground_truth['is_speaking']}, "
              f"MAR={ground_truth['mar']:.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    test_articulation_controller()
