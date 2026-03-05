"""
SMAIT HRI v2.0 - Metrics Collector and Analyzer

Collects test metrics and generates analysis reports.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class SingleMetric:
    """A single test measurement"""
    timestamp: float
    trial_id: int
    phase: str
    
    # Ground truth
    gt_speaking: bool
    gt_face_id: int = 0
    
    # Predictions
    pred_speaking: bool
    pred_confidence: float
    pred_face_id: int = 0
    
    # Timing
    latency_ms: float = 0.0
    
    # Parameters
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    @property
    def correct(self) -> bool:
        return self.gt_speaking == self.pred_speaking
    
    @property
    def is_tp(self) -> bool:
        return self.gt_speaking and self.pred_speaking
    
    @property
    def is_tn(self) -> bool:
        return not self.gt_speaking and not self.pred_speaking
    
    @property
    def is_fp(self) -> bool:
        return not self.gt_speaking and self.pred_speaking
    
    @property
    def is_fn(self) -> bool:
        return self.gt_speaking and not self.pred_speaking


class MetricsCollector:
    """
    Collects metrics during testing and provides analysis.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.metrics: List[SingleMetric] = []
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._start_time = time.time()
        self._current_phase = "unknown"
    
    def set_phase(self, phase: str):
        """Set current test phase"""
        self._current_phase = phase
    
    def record(
        self,
        gt_speaking: bool,
        pred_speaking: bool,
        pred_confidence: float,
        trial_id: int = 0,
        latency_ms: float = 0.0,
        parameters: Dict = None,
        gt_face_id: int = 0,
        pred_face_id: int = 0
    ):
        """Record a single test result"""
        metric = SingleMetric(
            timestamp=time.time(),
            trial_id=trial_id,
            phase=self._current_phase,
            gt_speaking=gt_speaking,
            gt_face_id=gt_face_id,
            pred_speaking=pred_speaking,
            pred_confidence=pred_confidence,
            pred_face_id=pred_face_id,
            latency_ms=latency_ms,
            parameters=parameters or {}
        )
        self.metrics.append(metric)
        
        return metric.correct
    
    def get_accuracy(self, phase: Optional[str] = None) -> float:
        """Get accuracy for a phase or overall"""
        metrics = self._filter_phase(phase)
        if not metrics:
            return 0.0
        return sum(1 for m in metrics if m.correct) / len(metrics)
    
    def get_confusion_matrix(self, phase: Optional[str] = None) -> Dict[str, int]:
        """Get confusion matrix counts"""
        metrics = self._filter_phase(phase)
        return {
            'tp': sum(1 for m in metrics if m.is_tp),
            'tn': sum(1 for m in metrics if m.is_tn),
            'fp': sum(1 for m in metrics if m.is_fp),
            'fn': sum(1 for m in metrics if m.is_fn)
        }
    
    def get_precision(self, phase: Optional[str] = None) -> float:
        """Get precision (TP / (TP + FP))"""
        cm = self.get_confusion_matrix(phase)
        denom = cm['tp'] + cm['fp']
        return cm['tp'] / denom if denom > 0 else 0.0
    
    def get_recall(self, phase: Optional[str] = None) -> float:
        """Get recall (TP / (TP + FN))"""
        cm = self.get_confusion_matrix(phase)
        denom = cm['tp'] + cm['fn']
        return cm['tp'] / denom if denom > 0 else 0.0
    
    def get_f1(self, phase: Optional[str] = None) -> float:
        """Get F1 score"""
        p = self.get_precision(phase)
        r = self.get_recall(phase)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def get_latency_stats(self, phase: Optional[str] = None) -> Dict[str, float]:
        """Get latency statistics"""
        metrics = self._filter_phase(phase)
        latencies = [m.latency_ms for m in metrics if m.latency_ms > 0]
        
        if not latencies:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p95': 0}
        
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p95': np.percentile(latencies, 95)
        }
    
    def group_by_parameter(
        self, 
        param_name: str,
        phase: Optional[str] = None
    ) -> Dict[any, List[SingleMetric]]:
        """Group metrics by a parameter value"""
        metrics = self._filter_phase(phase)
        groups = defaultdict(list)
        
        for m in metrics:
            value = m.parameters.get(param_name)
            if value is not None:
                groups[value].append(m)
        
        return dict(groups)
    
    def analyze_parameter(
        self,
        param_name: str,
        phase: Optional[str] = None
    ) -> List[Dict]:
        """Analyze accuracy across parameter values"""
        groups = self.group_by_parameter(param_name, phase)
        
        results = []
        for value, metrics in sorted(groups.items()):
            acc = sum(1 for m in metrics if m.correct) / len(metrics)
            results.append({
                'parameter': param_name,
                'value': value,
                'accuracy': acc,
                'n_samples': len(metrics)
            })
        
        return results
    
    def find_best_parameters(
        self,
        param_names: List[str],
        phase: Optional[str] = None
    ) -> Dict:
        """Find parameter combination with best accuracy"""
        metrics = self._filter_phase(phase)
        
        # Group by parameter combination
        groups = defaultdict(list)
        for m in metrics:
            key = tuple(m.parameters.get(p) for p in param_names)
            groups[key].append(m)
        
        # Find best
        best_key = None
        best_acc = 0.0
        
        for key, group in groups.items():
            acc = sum(1 for m in group if m.correct) / len(group)
            if acc > best_acc:
                best_acc = acc
                best_key = key
        
        if best_key:
            return {
                'parameters': dict(zip(param_names, best_key)),
                'accuracy': best_acc,
                'n_samples': len(groups[best_key])
            }
        
        return {}
    
    def _filter_phase(self, phase: Optional[str]) -> List[SingleMetric]:
        """Filter metrics by phase"""
        if phase is None:
            return self.metrics
        return [m for m in self.metrics if m.phase == phase]
    
    def save_csv(self, filename: str = "metrics.csv"):
        """Save metrics to CSV"""
        path = self.output_dir / filename
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'trial_id', 'phase',
                'gt_speaking', 'pred_speaking', 'correct',
                'confidence', 'latency_ms', 'parameters'
            ])
            
            # Data
            for m in self.metrics:
                writer.writerow([
                    m.timestamp, m.trial_id, m.phase,
                    m.gt_speaking, m.pred_speaking, m.correct,
                    m.pred_confidence, m.latency_ms,
                    json.dumps(m.parameters)
                ])
        
        print(f"[METRICS] Saved to {path}")
        return path
    
    def save_summary(self, filename: str = "summary.json"):
        """Save analysis summary to JSON"""
        path = self.output_dir / filename
        
        # Get all phases
        phases = set(m.phase for m in self.metrics)
        
        summary = {
            'total_metrics': len(self.metrics),
            'duration_seconds': time.time() - self._start_time,
            'overall': {
                'accuracy': self.get_accuracy(),
                'precision': self.get_precision(),
                'recall': self.get_recall(),
                'f1': self.get_f1(),
                'confusion_matrix': self.get_confusion_matrix(),
                'latency': self.get_latency_stats()
            },
            'by_phase': {}
        }
        
        for phase in phases:
            summary['by_phase'][phase] = {
                'accuracy': self.get_accuracy(phase),
                'precision': self.get_precision(phase),
                'recall': self.get_recall(phase),
                'f1': self.get_f1(phase),
                'confusion_matrix': self.get_confusion_matrix(phase),
                'n_samples': len(self._filter_phase(phase))
            }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[METRICS] Summary saved to {path}")
        return summary
    
    def generate_report(self) -> str:
        """Generate human-readable report"""
        summary = self.save_summary()
        
        report = []
        report.append("=" * 60)
        report.append("SMAIT HRI v2.0 - Test Results Report")
        report.append("=" * 60)
        report.append("")
        
        # Overall
        report.append("OVERALL RESULTS")
        report.append("-" * 40)
        overall = summary['overall']
        report.append(f"Total Trials: {summary['total_metrics']}")
        report.append(f"Duration: {summary['duration_seconds']:.1f}s")
        report.append(f"Accuracy: {overall['accuracy']:.1%}")
        report.append(f"Precision: {overall['precision']:.1%}")
        report.append(f"Recall: {overall['recall']:.1%}")
        report.append(f"F1 Score: {overall['f1']:.1%}")
        report.append("")
        
        # Confusion matrix
        cm = overall['confusion_matrix']
        report.append("Confusion Matrix:")
        report.append(f"              Predicted")
        report.append(f"              Speaking  Silent")
        report.append(f"Actual Speaking   {cm['tp']:4d}    {cm['fn']:4d}")
        report.append(f"       Silent     {cm['fp']:4d}    {cm['tn']:4d}")
        report.append("")
        
        # Latency
        lat = overall['latency']
        report.append(f"Latency: {lat['mean']:.0f}ms ± {lat['std']:.0f}ms "
                     f"(p95: {lat['p95']:.0f}ms)")
        report.append("")
        
        # By phase
        report.append("RESULTS BY PHASE")
        report.append("-" * 40)
        for phase, data in summary['by_phase'].items():
            report.append(f"{phase}:")
            report.append(f"  Accuracy: {data['accuracy']:.1%} "
                         f"(n={data['n_samples']})")
            report.append(f"  Precision: {data['precision']:.1%}, "
                         f"Recall: {data['recall']:.1%}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        path = self.output_dir / "report.txt"
        with open(path, 'w') as f:
            f.write(report_text)
        
        return report_text


def test_metrics_collector():
    """Test the metrics collector"""
    import random
    
    print("Testing MetricsCollector")
    print("=" * 40)
    
    collector = MetricsCollector("./test_results")
    
    # Simulate Phase 1
    collector.set_phase("phase1_asd_tuning")
    
    for i in range(100):
        gt = random.random() > 0.5
        # Simulate 90% accuracy
        pred = gt if random.random() > 0.1 else not gt
        conf = random.uniform(0.5, 1.0) if pred else random.uniform(0.0, 0.5)
        
        collector.record(
            gt_speaking=gt,
            pred_speaking=pred,
            pred_confidence=conf,
            trial_id=i,
            latency_ms=random.uniform(50, 200),
            parameters={'threshold': random.choice([0.3, 0.4, 0.5])}
        )
    
    # Simulate Phase 2
    collector.set_phase("phase2_av_delay")
    
    for delay in [-200, -100, 0, 100, 200]:
        for i in range(20):
            gt = True
            # Accuracy degrades with larger delays
            acc_drop = abs(delay) / 500
            pred = gt if random.random() > (0.1 + acc_drop) else not gt
            
            collector.record(
                gt_speaking=gt,
                pred_speaking=pred,
                pred_confidence=random.uniform(0.4, 0.9),
                trial_id=i,
                latency_ms=random.uniform(50, 150),
                parameters={'av_delay_ms': delay}
            )
    
    # Generate report
    report = collector.generate_report()
    print(report)
    
    # Analyze delay parameter
    print("\nDelay Analysis:")
    analysis = collector.analyze_parameter('av_delay_ms', 'phase2_av_delay')
    for item in analysis:
        print(f"  Delay {item['value']:+d}ms: {item['accuracy']:.1%}")
    
    print("\nDone!")


if __name__ == "__main__":
    test_metrics_collector()
