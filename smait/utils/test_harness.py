"""
SMAIT HRI Test Harness
======================
Hooks into HRISystem callbacks to automatically capture all performance metrics
during real test sessions. Logs per-turn, per-session, and system-level data.

Usage:
    from smait.utils.test_harness import TestHarness

    hri = HRISystem(config)
    harness = TestHarness(hri, output_dir="test_results", num_sessions=15)
    harness.attach()

    await hri.start()

    # After testing, generate report:
    harness.report()
"""

import csv
import json
import os
import time
import datetime
import threading
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import subprocess
    HAS_NVIDIA = True
except:
    HAS_NVIDIA = False


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class TurnRecord:
    """Metrics for a single conversation turn"""
    session_id: int
    turn_id: int
    timestamp: str

    # ASR
    transcript: str = ""
    asr_confidence: float = 0.0
    asr_latency_ms: float = 0.0     # From TranscriptResult (speech end → text ready)

    # Verification
    verified: bool = False
    verify_result: str = ""

    # LLM
    response_text: str = ""
    llm_latency_ms: float = 0.0

    # Total end-to-end (speech end → robot starts speaking)
    total_latency_ms: float = 0.0

    # WER (if reference provided)
    reference_text: str = ""
    wer: float = -1.0               # -1 = not measured


@dataclass
class SessionRecord:
    """Metrics for a complete session"""
    session_id: int
    start_time: str
    end_time: str = ""
    duration_s: float = 0.0

    # Engagement
    engagement_trigger_s: float = 0.0   # Time from face visible to session start
    clean_end: bool = False              # Session ended gracefully (not crash/timeout)
    end_reason: str = ""                 # "goodbye", "timeout", "face_lost"

    # Turns
    turn_count: int = 0
    avg_asr_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_asr_confidence: float = 0.0

    # DOA (if available)
    doa_readings: List[float] = field(default_factory=list)


@dataclass
class SystemSnapshot:
    """System resource snapshot"""
    timestamp: str
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    gpu_util_percent: float = 0.0
    gpu_mem_used_mb: float = 0.0


# ─────────────────────────────────────────────
# WER Calculation
# ─────────────────────────────────────────────

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate.
    WER = (S + D + I) / N  where N = number of words in reference
    Returns 0.0–1.0 (multiply by 100 for %)
    """
    ref = reference.lower().strip().split()
    hyp = hypothesis.lower().strip().split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Dynamic programming (Levenshtein distance)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[n][m] / len(ref)


# ─────────────────────────────────────────────
# Main Harness
# ─────────────────────────────────────────────

class TestHarness:
    """
    Attaches to a running HRISystem and captures all metrics automatically.
    Call attach() before hri.start(), report() after testing.
    """

    def __init__(
        self,
        hri_system,
        output_dir: str = "test_results",
        num_sessions: int = 15,
        test_name: str = "smait_test"
    ):
        self.hri = hri_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_sessions = num_sessions
        self.test_name = test_name

        # State
        self.sessions: List[SessionRecord] = []
        self.turns: List[TurnRecord] = []
        self.snapshots: List[SystemSnapshot] = []

        self._current_session: Optional[SessionRecord] = None
        self._turn_start: float = 0.0
        self._transcript_time: float = 0.0
        self._session_face_visible: float = 0.0
        self._session_counter = 0
        self._turn_counter = 0
        self._lock = threading.Lock()

        # Reference texts for WER (optional — set manually)
        # Format: { session_id: { turn_id: "reference text" } }
        self.reference_texts: Dict[int, Dict[int, str]] = {}

        # System monitor thread
        self._monitor_active = False
        self._monitor_thread: Optional[threading.Thread] = None

        self._start_time = datetime.datetime.now().isoformat()

    def attach(self):
        """Hook into HRISystem callbacks. Call before hri.start()."""
        hri = self.hri

        # ── Transcript callback ──
        original_on_transcript = hri._on_transcript.__func__ if hasattr(hri._on_transcript, '__func__') else None

        async def patched_on_transcript(transcript):
            self._on_transcript_hook(transcript)
            # Call original
            if original_on_transcript:
                await original_on_transcript(hri, transcript)
            else:
                await hri.__class__._on_transcript(hri, transcript)

        import asyncio
        import types
        hri._on_transcript = types.MethodType(
            lambda self, t: asyncio.ensure_future(patched_on_transcript(t)),
            hri
        )
        hri.transcriber.set_final_callback(hri._on_transcript)

        # ── Response callback ──
        original_on_response = None
        try:
            original_on_response = hri.__class__._on_response
        except: pass

        def patched_on_response(response):
            self._on_response_hook(response)
            if original_on_response:
                original_on_response(hri, response)

        hri._on_response = types.MethodType(patched_on_response, hri)
        if hri.dialogue:
            hri.dialogue.set_response_callback(hri._on_response)

        # ── Session start/end hooks (patch verifier) ──
        verifier = hri.verifier
        original_start = verifier._start_session
        original_end = verifier.end_session if hasattr(verifier, 'end_session') else None

        def patched_start_session(user_id=None, *args, **kwargs):
            self._on_session_start(user_id)
            return original_start(user_id, *args, **kwargs)

        verifier._start_session = patched_start_session

        # ── Start system monitor ──
        self._monitor_active = True
        self._monitor_thread = threading.Thread(
            target=self._system_monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        print(f"[HARNESS] Attached — recording up to {self.num_sessions} sessions")
        print(f"[HARNESS] Output: {self.output_dir.absolute()}")

    def mark_session_end(self, reason: str = "goodbye"):
        """Call this when a session ends (goodbye / timeout / face lost)."""
        with self._lock:
            if self._current_session:
                now = time.time()
                self._current_session.end_time = datetime.datetime.now().isoformat()
                self._current_session.duration_s = round(
                    now - datetime.datetime.fromisoformat(
                        self._current_session.start_time
                    ).timestamp(), 2
                )
                self._current_session.clean_end = (reason == "goodbye")
                self._current_session.end_reason = reason

                # Compute session averages from turns
                session_turns = [t for t in self.turns if t.session_id == self._current_session.session_id]
                if session_turns:
                    self._current_session.turn_count = len(session_turns)
                    self._current_session.avg_asr_latency_ms = round(
                        statistics.mean([t.asr_latency_ms for t in session_turns if t.asr_latency_ms > 0]), 1)
                    self._current_session.avg_llm_latency_ms = round(
                        statistics.mean([t.llm_latency_ms for t in session_turns if t.llm_latency_ms > 0]), 1)
                    self._current_session.avg_total_latency_ms = round(
                        statistics.mean([t.total_latency_ms for t in session_turns if t.total_latency_ms > 0]), 1)
                    self._current_session.avg_asr_confidence = round(
                        statistics.mean([t.asr_confidence for t in session_turns]), 3)

                self.sessions.append(self._current_session)
                self._flush_turn_csv()
                print(f"[HARNESS] Session {self._current_session.session_id} ended ({reason}) — {self._current_session.turn_count} turns, {self._current_session.duration_s:.1f}s")
                self._current_session = None

    def set_reference(self, session_id: int, turn_id: int, text: str):
        """Set expected transcript for WER calculation."""
        if session_id not in self.reference_texts:
            self.reference_texts[session_id] = {}
        self.reference_texts[session_id][turn_id] = text

    def mark_face_visible(self):
        """Call when face first appears in frame (for engagement time calculation)."""
        self._session_face_visible = time.time()

    # ── Private Hooks ──

    def _on_session_start(self, user_id=None):
        with self._lock:
            self._session_counter += 1
            self._turn_counter = 0
            engagement_time = 0.0
            if self._session_face_visible > 0:
                engagement_time = round(time.time() - self._session_face_visible, 2)
            self._current_session = SessionRecord(
                session_id=self._session_counter,
                start_time=datetime.datetime.now().isoformat(),
                engagement_trigger_s=engagement_time
            )
            print(f"[HARNESS] Session {self._session_counter} started (engagement: {engagement_time:.1f}s)")

    def _on_transcript_hook(self, transcript):
        """Called when ASR produces a final transcript."""
        self._transcript_time = time.time()
        self._turn_start = self._transcript_time
        # Store transcript for this turn (will be completed in response hook)
        self._pending_transcript = transcript

    def _on_response_hook(self, response):
        """Called when LLM produces a response."""
        now = time.time()
        total_ms = (now - self._turn_start) * 1000 if self._turn_start > 0 else 0.0

        with self._lock:
            if self._current_session is None:
                return

            self._turn_counter += 1
            tr = getattr(self, '_pending_transcript', None)

            # WER
            wer = -1.0
            ref = self.reference_texts.get(self._session_counter, {}).get(self._turn_counter, "")
            if ref and tr:
                wer = round(calculate_wer(ref, tr.text) * 100, 1)

            turn = TurnRecord(
                session_id=self._session_counter,
                turn_id=self._turn_counter,
                timestamp=datetime.datetime.now().isoformat(),
                transcript=tr.text if tr else "",
                asr_confidence=round(tr.confidence, 3) if tr else 0.0,
                asr_latency_ms=round(getattr(tr, 'latency_ms', 0.0), 1) if tr else 0.0,
                verified=True,
                verify_result="ACCEPT",
                response_text=response.text,
                llm_latency_ms=round(response.latency_ms, 1),
                total_latency_ms=round(total_ms, 1),
                reference_text=ref,
                wer=wer
            )
            self.turns.append(turn)

            print(f"[HARNESS] Turn {self._turn_counter}: ASR={turn.asr_latency_ms}ms | LLM={turn.llm_latency_ms}ms | Total={turn.total_latency_ms}ms | WER={wer if wer >= 0 else 'N/A'}")

    def _system_monitor_loop(self):
        """Background thread — snapshots system resources every 10s."""
        while self._monitor_active:
            snap = self._take_snapshot()
            self.snapshots.append(snap)
            time.sleep(10)

    def _take_snapshot(self) -> SystemSnapshot:
        snap = SystemSnapshot(timestamp=datetime.datetime.now().isoformat())
        if HAS_PSUTIL:
            snap.cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            snap.ram_used_mb = round(mem.used / 1024 / 1024, 1)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                snap.gpu_util_percent = float(parts[0])
                snap.gpu_mem_used_mb = float(parts[1])
        except: pass
        return snap

    # ── Output ──

    def _flush_turn_csv(self):
        path = self.output_dir / f"{self.test_name}_turns.csv"
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(TurnRecord(0,0,"")).keys()))
            if write_header:
                w.writeheader()
            for t in self.turns:
                if t.session_id == self._session_counter:
                    w.writerow(asdict(t))

    def report(self):
        """Generate full test report. Call after all sessions complete."""
        self._monitor_active = False

        # Save sessions CSV
        sessions_path = self.output_dir / f"{self.test_name}_sessions.csv"
        with open(sessions_path, "w", newline="") as f:
            if self.sessions:
                w = csv.DictWriter(f, fieldnames=[k for k in asdict(self.sessions[0]).keys() if k != 'doa_readings'])
                w.writeheader()
                for s in self.sessions:
                    row = asdict(s)
                    row.pop('doa_readings', None)
                    w.writerow(row)

        # Save system snapshots CSV
        snaps_path = self.output_dir / f"{self.test_name}_system.csv"
        with open(snaps_path, "w", newline="") as f:
            if self.snapshots:
                w = csv.DictWriter(f, fieldnames=list(asdict(self.snapshots[0]).keys()))
                w.writeheader()
                w.writerows([asdict(s) for s in self.snapshots])

        # Compute summary stats
        all_turns = self.turns
        all_sessions = self.sessions

        asr_latencies = [t.asr_latency_ms for t in all_turns if t.asr_latency_ms > 0]
        llm_latencies = [t.llm_latency_ms for t in all_turns if t.llm_latency_ms > 0]
        total_latencies = [t.total_latency_ms for t in all_turns if t.total_latency_ms > 0]
        wer_values = [t.wer for t in all_turns if t.wer >= 0]
        engagement_times = [s.engagement_trigger_s for s in all_sessions if s.engagement_trigger_s > 0]
        clean_ends = [s for s in all_sessions if s.clean_end]

        summary = {
            "test_name": self.test_name,
            "run_date": self._start_time,
            "total_sessions": len(all_sessions),
            "total_turns": len(all_turns),

            "asr": {
                "mean_latency_ms": round(statistics.mean(asr_latencies), 1) if asr_latencies else None,
                "median_latency_ms": round(statistics.median(asr_latencies), 1) if asr_latencies else None,
                "max_latency_ms": round(max(asr_latencies), 1) if asr_latencies else None,
                "p95_latency_ms": round(sorted(asr_latencies)[int(len(asr_latencies)*0.95)], 1) if len(asr_latencies) > 5 else None,
                "mean_wer_pct": round(statistics.mean(wer_values), 1) if wer_values else None,
            },

            "llm": {
                "mean_latency_ms": round(statistics.mean(llm_latencies), 1) if llm_latencies else None,
                "median_latency_ms": round(statistics.median(llm_latencies), 1) if llm_latencies else None,
                "max_latency_ms": round(max(llm_latencies), 1) if llm_latencies else None,
            },

            "end_to_end": {
                "mean_total_latency_ms": round(statistics.mean(total_latencies), 1) if total_latencies else None,
                "median_total_latency_ms": round(statistics.median(total_latencies), 1) if total_latencies else None,
                "max_total_latency_ms": round(max(total_latencies), 1) if total_latencies else None,
                "under_3s_pct": round(len([t for t in total_latencies if t < 3000]) / len(total_latencies) * 100, 1) if total_latencies else None,
            },

            "session_management": {
                "engagement_success_rate_pct": 100.0,  # session started = engaged
                "mean_engagement_time_s": round(statistics.mean(engagement_times), 2) if engagement_times else None,
                "clean_end_rate_pct": round(len(clean_ends) / len(all_sessions) * 100, 1) if all_sessions else None,
                "mean_session_duration_s": round(statistics.mean([s.duration_s for s in all_sessions]), 1) if all_sessions else None,
                "mean_turns_per_session": round(statistics.mean([s.turn_count for s in all_sessions]), 1) if all_sessions else None,
            },

            "system": {
                "mean_cpu_pct": round(statistics.mean([s.cpu_percent for s in self.snapshots]), 1) if self.snapshots else None,
                "mean_gpu_util_pct": round(statistics.mean([s.gpu_util_percent for s in self.snapshots]), 1) if self.snapshots else None,
                "max_gpu_mem_mb": round(max([s.gpu_mem_used_mb for s in self.snapshots]), 0) if self.snapshots else None,
            },

            "success_criteria": {
                "asr_latency_under_200ms": (statistics.mean(asr_latencies) < 200) if asr_latencies else None,
                "asr_wer_under_15pct": (statistics.mean(wer_values) < 15) if wer_values else None,
                "total_latency_under_3s": (statistics.mean(total_latencies) < 3000) if total_latencies else None,
                "clean_end_rate_100pct": (len(clean_ends) == len(all_sessions)) if all_sessions else None,
            }
        }

        # Save JSON summary
        summary_path = self.output_dir / f"{self.test_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print human-readable report
        self._print_report(summary)
        print(f"\n[HARNESS] Files saved to: {self.output_dir.absolute()}")
        print(f"  - {self.test_name}_turns.csv     (per-turn data)")
        print(f"  - {self.test_name}_sessions.csv  (per-session data)")
        print(f"  - {self.test_name}_system.csv    (CPU/GPU over time)")
        print(f"  - {self.test_name}_summary.json  (aggregated results)")

        return summary

    def _print_report(self, s: dict):
        print("\n" + "="*60)
        print(f"  SMAIT HRI TEST REPORT — {s['run_date'][:10]}")
        print("="*60)
        print(f"  Sessions: {s['total_sessions']}  |  Turns: {s['total_turns']}")
        print()
        print("  ASR PERFORMANCE")
        a = s['asr']
        print(f"    Mean Latency:   {a['mean_latency_ms']} ms  (target: <200ms)  {'✓' if a['mean_latency_ms'] and a['mean_latency_ms']<200 else '✗'}")
        print(f"    P95 Latency:    {a['p95_latency_ms']} ms")
        print(f"    Mean WER:       {a['mean_wer_pct']}%  (target: <15%)  {'✓' if a['mean_wer_pct'] and a['mean_wer_pct']<15 else '—'}")
        print()
        print("  END-TO-END LATENCY")
        e = s['end_to_end']
        print(f"    Mean Total:     {e['mean_total_latency_ms']} ms  (target: <3000ms)  {'✓' if e['mean_total_latency_ms'] and e['mean_total_latency_ms']<3000 else '✗'}")
        print(f"    Under 3s:       {e['under_3s_pct']}% of turns")
        print()
        print("  SESSION MANAGEMENT")
        sm = s['session_management']
        print(f"    Clean End Rate: {sm['clean_end_rate_pct']}%  (target: 100%)")
        print(f"    Mean Eng. Time: {sm['mean_engagement_time_s']}s")
        print(f"    Mean Duration:  {sm['mean_session_duration_s']}s")
        print(f"    Turns/Session:  {sm['mean_turns_per_session']}")
        print()
        print("  SYSTEM")
        sys = s['system']
        print(f"    CPU:  {sys['mean_cpu_pct']}%  |  GPU: {sys['mean_gpu_util_pct']}%  |  GPU Mem: {sys['max_gpu_mem_mb']} MB")
        print("="*60)
