"""Latency/performance metrics capture."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

MAX_HISTORY = 100  # Keep last N measurements per metric


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    duration_ms: float
    timestamp: float


class MetricsTracker:
    """Capture latency and performance metrics across the pipeline.

    Tracks per-component timing with rolling averages and percentiles.
    """

    def __init__(self) -> None:
        self._timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self._counters: dict[str, int] = defaultdict(int)
        self._active_timers: dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._active_timers[name] = time.monotonic()

    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a named timer and record the duration.

        Returns duration in milliseconds, or None if timer wasn't started.
        """
        start = self._active_timers.pop(name, None)
        if start is None:
            return None

        duration_ms = (time.monotonic() - start) * 1000
        self._timings[name].append(duration_ms)
        return duration_ms

    def record(self, name: str, duration_ms: float) -> None:
        """Directly record a timing measurement."""
        self._timings[name].append(duration_ms)

    def increment(self, name: str, count: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] += count

    def get_average(self, name: str) -> Optional[float]:
        """Get rolling average for a metric in ms."""
        values = self._timings.get(name)
        if not values:
            return None
        return sum(values) / len(values)

    def get_p95(self, name: str) -> Optional[float]:
        """Get 95th percentile for a metric in ms."""
        values = self._timings.get(name)
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def get_count(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)

    def get_summary(self) -> dict[str, dict]:
        """Get a summary of all metrics."""
        summary = {}

        for name, values in self._timings.items():
            if values:
                sorted_vals = sorted(values)
                summary[name] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "min_ms": sorted_vals[0],
                    "max_ms": sorted_vals[-1],
                    "p50_ms": sorted_vals[len(sorted_vals) // 2],
                    "p95_ms": sorted_vals[int(len(sorted_vals) * 0.95)],
                }

        for name, count in self._counters.items():
            if name not in summary:
                summary[name] = {}
            summary[name]["total_count"] = count

        return summary

    def log_summary(self) -> None:
        """Log a summary of all metrics."""
        summary = self.get_summary()
        if not summary:
            return

        logger.info("=== Performance Metrics ===")
        for name, stats in sorted(summary.items()):
            if "avg_ms" in stats:
                logger.info("  %s: avg=%.1fms p95=%.1fms (n=%d)",
                            name, stats["avg_ms"], stats.get("p95_ms", 0), stats["count"])
            elif "total_count" in stats:
                logger.info("  %s: count=%d", name, stats["total_count"])
        logger.info("===========================")

    def reset(self) -> None:
        """Reset all metrics."""
        self._timings.clear()
        self._counters.clear()
        self._active_timers.clear()
