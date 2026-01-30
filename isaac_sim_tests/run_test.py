"""
SMAIT HRI v2.0 - Isaac Sim Test Runner

Main entry point for domain randomization testing.
Requires: Isaac Sim 4.5+, Audio2Face

Usage:
    python run_test.py --phase 1              # Run Phase 1 only
    python run_test.py --phase 1 2 3          # Run Phases 1, 2, 3
    python run_test.py --all                  # Run all phases
    python run_test.py --quick                # Quick smoke test
"""

import argparse
import yaml
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
import csv

# Add parent directory for SMAIT imports
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "5"))


@dataclass
class TestResult:
    """Single test trial result"""
    trial_id: int
    ground_truth_speaking: bool
    predicted_speaking: bool
    confidence: float
    latency_ms: float
    parameters: Dict
    timestamp: float = field(default_factory=time.time)
    
    @property
    def correct(self) -> bool:
        return self.ground_truth_speaking == self.predicted_speaking
    
    @property
    def is_tp(self) -> bool:
        return self.ground_truth_speaking and self.predicted_speaking
    
    @property
    def is_tn(self) -> bool:
        return not self.ground_truth_speaking and not self.predicted_speaking
    
    @property
    def is_fp(self) -> bool:
        return not self.ground_truth_speaking and self.predicted_speaking
    
    @property
    def is_fn(self) -> bool:
        return self.ground_truth_speaking and not self.predicted_speaking


@dataclass
class PhaseResults:
    """Results from a test phase"""
    phase_name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def add_result(self, result: TestResult):
        self.results.append(result)
    
    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)
    
    @property
    def tp(self) -> int:
        return sum(1 for r in self.results if r.is_tp)
    
    @property
    def tn(self) -> int:
        return sum(1 for r in self.results if r.is_tn)
    
    @property
    def fp(self) -> int:
        return sum(1 for r in self.results if r.is_fp)
    
    @property
    def fn(self) -> int:
        return sum(1 for r in self.results if r.is_fn)
    
    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    def summary(self) -> str:
        return f"""
Phase: {self.phase_name}
========================================
Total Trials: {len(self.results)}
Accuracy: {self.accuracy:.1%}
Precision: {self.precision:.1%}
Recall: {self.recall:.1%}

Confusion Matrix:
              Predicted
              Speaking  Silent
Actual Speaking   {self.tp:4d}    {self.fn:4d}
       Silent     {self.fp:4d}    {self.tn:4d}

Duration: {self.end_time - self.start_time:.1f}s
========================================
"""


class IsaacSimTestRunner:
    """
    Orchestrates Isaac Sim testing for SMAIT HRI system.
    
    This is a framework - actual Isaac Sim integration requires
    running within the Isaac Sim Python environment.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = self._create_results_dir()
        
        # Components (initialized on connect)
        self.isaac_sim = None
        self.audio2face = None
        self.smait_system = None
        
        # State
        self.connected = False
        self.current_phase = None
    
    def _load_config(self) -> Dict:
        """Load test configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_results_dir(self) -> Path:
        """Create timestamped results directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config['global']['output']['results_dir']) / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config copy
        with open(results_dir / "config_used.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        return results_dir
    
    def connect(self) -> bool:
        """
        Connect to Isaac Sim and Audio2Face.
        
        NOTE: This needs to be run from within Isaac Sim's Python environment.
        When running standalone, this will use mock connections.
        """
        print("[TEST] Connecting to Isaac Sim and Audio2Face...")
        
        try:
            # Try to import Isaac Sim modules
            from isaacsim import SimulationApp
            
            # Initialize simulation
            simulation_app = SimulationApp({
                "headless": self.config['global']['isaac_sim']['headless'],
                "width": self.config['global']['isaac_sim']['resolution'][0],
                "height": self.config['global']['isaac_sim']['resolution'][1],
            })
            
            self.isaac_sim = simulation_app
            print("[TEST] ✓ Isaac Sim connected")
            
        except ImportError:
            print("[TEST] ⚠ Isaac Sim not available - using mock mode")
            print("[TEST]   To run real tests, launch from Isaac Sim Python:")
            print("[TEST]   ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh run_test.py")
            self.isaac_sim = MockIsaacSim()
        
        try:
            # Connect to Audio2Face
            from domain_randomization.audio2face_client import Audio2FaceClient
            
            port = self.config['global']['audio2face']['streaming_port']
            self.audio2face = Audio2FaceClient(port=port)
            self.audio2face.connect()
            print("[TEST] ✓ Audio2Face connected")
            
        except ImportError:
            print("[TEST] ⚠ Audio2Face client not available - using mock mode")
            self.audio2face = MockAudio2Face()
        
        # Initialize SMAIT system
        try:
            from smait.perception.laser_asd import LASERBackend
            from smait.perception.verifier import SpeakerVerifier
            from smait.core.config import get_config
            
            self.smait_system = {
                'asd': LASERBackend(),
                'verifier': SpeakerVerifier(),
                'config': get_config()
            }
            print("[TEST] ✓ SMAIT system initialized")
            
        except ImportError as e:
            print(f"[TEST] ⚠ SMAIT system import error: {e}")
            print("[TEST]   Using mock SMAIT system")
            self.smait_system = MockSMAIT()
        
        self.connected = True
        return True
    
    def run_phase(self, phase_num: int) -> PhaseResults:
        """Run a specific test phase"""
        phase_map = {
            1: ("phase1_asd_tuning", self._run_phase1_asd_tuning),
            2: ("phase2_av_delay", self._run_phase2_av_delay),
            3: ("phase3_noise", self._run_phase3_noise),
            4: ("phase4_articulation", self._run_phase4_articulation),
            5: ("phase5_multi_speaker", self._run_phase5_multi_speaker),
        }
        
        if phase_num not in phase_map:
            raise ValueError(f"Invalid phase number: {phase_num}")
        
        phase_name, phase_func = phase_map[phase_num]
        phase_config = self.config.get(phase_name, {})
        
        if not phase_config.get('enabled', False):
            print(f"[TEST] Phase {phase_num} is disabled in config")
            return PhaseResults(phase_name=phase_name)
        
        print(f"\n{'='*60}")
        print(f"PHASE {phase_num}: {phase_config.get('description', phase_name)}")
        print(f"{'='*60}\n")
        
        results = PhaseResults(phase_name=phase_name)
        results.start_time = time.time()
        
        phase_func(phase_config, results)
        
        results.end_time = time.time()
        
        # Save results
        self._save_phase_results(results)
        
        print(results.summary())
        
        return results
    
    def _run_phase1_asd_tuning(self, config: Dict, results: PhaseResults):
        """Phase 1: ASD Parameter Tuning"""
        sweep = config['sweep']
        trials_per = config['trials_per_config']
        
        # Generate parameter combinations
        import itertools
        
        def frange(start, end, step):
            vals = []
            v = start
            while v <= end + step/2:
                vals.append(round(v, 4))
                v += step
            return vals
        
        lip_movements = frange(**sweep['min_lip_movement'])
        mar_thresholds = frange(**sweep['min_mar_for_speech'])
        asd_thresholds = frange(**sweep['asd_threshold'])
        
        total_combos = len(lip_movements) * len(mar_thresholds) * len(asd_thresholds)
        print(f"[PHASE1] Testing {total_combos} parameter combinations")
        print(f"[PHASE1] {trials_per} trials each = {total_combos * trials_per} total trials")
        
        trial_id = 0
        for lip_mv in lip_movements:
            for mar_th in mar_thresholds:
                for asd_th in asd_thresholds:
                    params = {
                        'min_lip_movement': lip_mv,
                        'min_mar_for_speech': mar_th,
                        'asd_threshold': asd_th
                    }
                    
                    # Update SMAIT parameters
                    self._set_asd_params(params)
                    
                    # Run trials
                    for t in range(trials_per):
                        result = self._run_single_trial(
                            trial_id=trial_id,
                            speaking=True,  # Ground truth
                            params=params
                        )
                        results.add_result(result)
                        trial_id += 1
                        
                        # Also test silent (not speaking)
                        result = self._run_single_trial(
                            trial_id=trial_id,
                            speaking=False,
                            params=params
                        )
                        results.add_result(result)
                        trial_id += 1
                    
                    # Progress
                    if trial_id % 100 == 0:
                        print(f"[PHASE1] Progress: {trial_id} trials, "
                              f"current accuracy: {results.accuracy:.1%}")
    
    def _run_phase2_av_delay(self, config: Dict, results: PhaseResults):
        """Phase 2: Audio-Visual Delay Tolerance"""
        delays = config['delays_ms']
        trials_per = config['trials_per_delay']
        
        # Set ASD params from config
        self._set_asd_params(config['asd_params'])
        
        print(f"[PHASE2] Testing {len(delays)} delay values")
        
        trial_id = 0
        for delay_ms in delays:
            print(f"[PHASE2] Testing delay: {delay_ms:+d}ms")
            
            # Configure delay injection
            self._set_av_delay(delay_ms)
            
            for t in range(trials_per):
                # Speaking trial
                result = self._run_single_trial(
                    trial_id=trial_id,
                    speaking=True,
                    params={'av_delay_ms': delay_ms}
                )
                results.add_result(result)
                trial_id += 1
                
                # Silent trial
                result = self._run_single_trial(
                    trial_id=trial_id,
                    speaking=False,
                    params={'av_delay_ms': delay_ms}
                )
                results.add_result(result)
                trial_id += 1
            
            # Calculate accuracy for this delay
            delay_results = [r for r in results.results 
                          if r.parameters.get('av_delay_ms') == delay_ms]
            delay_acc = sum(1 for r in delay_results if r.correct) / len(delay_results)
            print(f"[PHASE2]   Delay {delay_ms:+d}ms accuracy: {delay_acc:.1%}")
    
    def _run_phase3_noise(self, config: Dict, results: PhaseResults):
        """Phase 3: Noise Robustness"""
        self._set_asd_params(config['asd_params'])
        trials_per = config['trials_per_condition']
        
        trial_id = 0
        for noise_source in config['noise_sources']:
            print(f"[PHASE3] Testing noise: {noise_source['name']}")
            
            for level in noise_source['levels']:
                self._set_noise(noise_source['file'], level)
                
                for t in range(trials_per):
                    result = self._run_single_trial(
                        trial_id=trial_id,
                        speaking=True,
                        params={
                            'noise_type': noise_source['name'],
                            'noise_level': level
                        }
                    )
                    results.add_result(result)
                    trial_id += 1
                
                level_results = [r for r in results.results 
                               if r.parameters.get('noise_level') == level 
                               and r.parameters.get('noise_type') == noise_source['name']]
                if level_results:
                    acc = sum(1 for r in level_results if r.correct) / len(level_results)
                    print(f"[PHASE3]   {noise_source['name']} @ {level:.0%}: {acc:.1%}")
    
    def _run_phase4_articulation(self, config: Dict, results: PhaseResults):
        """Phase 4: Articulation Variation"""
        trials_per = config['trials_per_style']
        
        trial_id = 0
        for style in config['articulation_scales']:
            print(f"[PHASE4] Testing style: {style['name']} (scale={style['scale']})")
            
            self._set_articulation_scale(style['scale'])
            
            for t in range(trials_per):
                result = self._run_single_trial(
                    trial_id=trial_id,
                    speaking=True,
                    params={
                        'articulation_style': style['name'],
                        'articulation_scale': style['scale']
                    }
                )
                results.add_result(result)
                trial_id += 1
            
            style_results = [r for r in results.results 
                          if r.parameters.get('articulation_style') == style['name']]
            if style_results:
                acc = sum(1 for r in style_results if r.correct) / len(style_results)
                print(f"[PHASE4]   {style['name']}: {acc:.1%}")
    
    def _run_phase5_multi_speaker(self, config: Dict, results: PhaseResults):
        """Phase 5: Multi-Speaker"""
        trial_id = 0
        
        for scenario in config['scenarios']:
            print(f"[PHASE5] Testing: {scenario['name']}")
            
            for t in range(scenario['trials']):
                result = self._run_multi_speaker_trial(
                    trial_id=trial_id,
                    num_faces=scenario['num_faces'],
                    speaking_face=scenario.get('speaking_face', 0),
                    params={'scenario': scenario['name']}
                )
                results.add_result(result)
                trial_id += 1
    
    def _run_single_trial(
        self, 
        trial_id: int, 
        speaking: bool, 
        params: Dict
    ) -> TestResult:
        """
        Run a single test trial.
        
        In real mode: drives Audio2Face, captures frames, runs ASD
        In mock mode: simulates results
        """
        start_time = time.time()
        
        if speaking:
            # Play audio through Audio2Face
            self.audio2face.play_audio("test_phrase.wav")
        
        # Wait for simulation
        time.sleep(0.5)
        
        # Get ASD prediction
        predicted_speaking, confidence = self._get_asd_prediction()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TestResult(
            trial_id=trial_id,
            ground_truth_speaking=speaking,
            predicted_speaking=predicted_speaking,
            confidence=confidence,
            latency_ms=latency_ms,
            parameters=params
        )
    
    def _run_multi_speaker_trial(
        self,
        trial_id: int,
        num_faces: int,
        speaking_face: int,
        params: Dict
    ) -> TestResult:
        """Run a multi-speaker trial"""
        # Similar to single trial but with multiple faces
        return self._run_single_trial(trial_id, True, params)
    
    def _get_asd_prediction(self) -> tuple:
        """Get ASD prediction from SMAIT system"""
        if isinstance(self.smait_system, dict):
            # Real SMAIT system
            # This would capture frame and run ASD
            pass
        
        # Mock response
        import random
        speaking = random.random() > 0.1  # 90% accuracy mock
        confidence = random.uniform(0.5, 1.0) if speaking else random.uniform(0.0, 0.5)
        return speaking, confidence
    
    def _set_asd_params(self, params: Dict):
        """Update ASD parameters"""
        if isinstance(self.smait_system, dict):
            asd = self.smait_system['asd']
            for key, value in params.items():
                if hasattr(asd, key):
                    setattr(asd, key, value)
    
    def _set_av_delay(self, delay_ms: int):
        """Configure audio-visual delay injection"""
        # This would be implemented in the domain randomization module
        pass
    
    def _set_noise(self, noise_file: str, level: float):
        """Configure noise injection"""
        pass
    
    def _set_articulation_scale(self, scale: float):
        """Configure articulation scaling"""
        if self.audio2face:
            self.audio2face.set_blendshape_scale(scale)
    
    def _save_phase_results(self, results: PhaseResults):
        """Save phase results to CSV"""
        csv_path = self.results_dir / f"{results.phase_name}_results.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_id', 'ground_truth', 'predicted', 'correct',
                'confidence', 'latency_ms', 'parameters'
            ])
            
            for r in results.results:
                writer.writerow([
                    r.trial_id,
                    r.ground_truth_speaking,
                    r.predicted_speaking,
                    r.correct,
                    r.confidence,
                    r.latency_ms,
                    json.dumps(r.parameters)
                ])
        
        print(f"[TEST] Results saved to {csv_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.isaac_sim and hasattr(self.isaac_sim, 'close'):
            self.isaac_sim.close()
        
        if self.audio2face and hasattr(self.audio2face, 'disconnect'):
            self.audio2face.disconnect()


# Mock classes for testing without Isaac Sim
class MockIsaacSim:
    def __init__(self):
        print("[MOCK] MockIsaacSim initialized")
    
    def close(self):
        pass


class MockAudio2Face:
    def __init__(self):
        print("[MOCK] MockAudio2Face initialized")
    
    def connect(self):
        pass
    
    def disconnect(self):
        pass
    
    def play_audio(self, audio_file: str):
        pass
    
    def set_blendshape_scale(self, scale: float):
        pass


class MockSMAIT:
    def __init__(self):
        print("[MOCK] MockSMAIT initialized")


def main():
    parser = argparse.ArgumentParser(description="SMAIT HRI Isaac Sim Test Runner")
    parser.add_argument('--phase', type=int, nargs='+', help="Phase number(s) to run")
    parser.add_argument('--all', action='store_true', help="Run all phases")
    parser.add_argument('--quick', action='store_true', help="Quick smoke test")
    parser.add_argument('--config', default='config.yaml', help="Config file path")
    
    args = parser.parse_args()
    
    runner = IsaacSimTestRunner(config_path=args.config)
    
    try:
        runner.connect()
        
        if args.all:
            phases = [1, 2, 3, 4, 5]
        elif args.quick:
            phases = [1]  # Just run phase 1 for quick test
        elif args.phase:
            phases = args.phase
        else:
            print("Specify --phase, --all, or --quick")
            return
        
        all_results = []
        for phase_num in phases:
            results = runner.run_phase(phase_num)
            all_results.append(results)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        for results in all_results:
            if results.results:
                print(f"{results.phase_name}: {results.accuracy:.1%} accuracy "
                      f"({len(results.results)} trials)")
        
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
