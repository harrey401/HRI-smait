#!/usr/bin/env python3
"""
SMAIT Metrics Tracker
=====================
Track test results and parameter changes over time.
Uses SQLite for persistence.

Usage:
    python track_metrics.py log --accuracy 0.93 --latency 700 --notes "Quiet room test"
    python track_metrics.py params --lip 0.008 --mar 0.05 --threshold 0.5
    python track_metrics.py trends
    python track_metrics.py export
"""

import sqlite3
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Database location
DB_PATH = Path(__file__).parent.parent / "data" / "smait_metrics.db"


def get_connection():
    """Get database connection, creating tables if needed"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    # Create tables
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS test_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            accuracy REAL,
            latency_ms REAL,
            false_positives INTEGER DEFAULT 0,
            false_negatives INTEGER DEFAULT 0,
            environment TEXT,
            config_snapshot TEXT,
            notes TEXT
        );
        
        CREATE TABLE IF NOT EXISTS parameter_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            min_lip_movement REAL,
            min_mar_for_speech REAL,
            asd_threshold REAL,
            smoothing_window INTEGER,
            reason TEXT
        );
        
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            duration_seconds REAL,
            total_utterances INTEGER,
            successful_detections INTEGER,
            session_type TEXT,
            recording_path TEXT
        );
    ''')
    conn.commit()
    return conn


def log_test_run(accuracy, latency, fp=0, fn=0, env="unknown", notes=""):
    """Log a test run result"""
    conn = get_connection()
    
    # Get current config
    config = get_current_config()
    
    conn.execute('''
        INSERT INTO test_runs 
        (timestamp, accuracy, latency_ms, false_positives, false_negatives, 
         environment, config_snapshot, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        accuracy,
        latency,
        fp,
        fn,
        env,
        json.dumps(config),
        notes
    ))
    conn.commit()
    
    print(f"✅ Logged test run:")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Latency: {latency}ms")
    if fp or fn:
        print(f"   FP: {fp}, FN: {fn}")


def log_parameter_change(lip=None, mar=None, threshold=None, window=None, reason=""):
    """Log a parameter change"""
    conn = get_connection()
    
    conn.execute('''
        INSERT INTO parameter_changes
        (timestamp, min_lip_movement, min_mar_for_speech, asd_threshold, 
         smoothing_window, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        lip,
        mar,
        threshold,
        window,
        reason
    ))
    conn.commit()
    
    print(f"✅ Logged parameter change:")
    if lip: print(f"   min_lip_movement: {lip}")
    if mar: print(f"   min_mar_for_speech: {mar}")
    if threshold: print(f"   asd_threshold: {threshold}")
    if window: print(f"   smoothing_window: {window}")
    if reason: print(f"   Reason: {reason}")


def show_trends(days=7):
    """Show trends over recent days"""
    conn = get_connection()
    
    # Test run trends
    print(f"\n📊 Test Run Trends (Last {days} days)")
    print("=" * 50)
    
    results = conn.execute('''
        SELECT 
            date(timestamp) as date,
            COUNT(*) as runs,
            AVG(accuracy) as avg_accuracy,
            AVG(latency_ms) as avg_latency,
            SUM(false_positives) as total_fp,
            SUM(false_negatives) as total_fn
        FROM test_runs
        WHERE timestamp > datetime('now', ?)
        GROUP BY date(timestamp)
        ORDER BY date(timestamp)
    ''', (f'-{days} days',)).fetchall()
    
    if not results:
        print("No test runs in this period.")
    else:
        print(f"{'Date':<12} {'Runs':<6} {'Accuracy':<10} {'Latency':<10} {'FP':<5} {'FN':<5}")
        print("-" * 50)
        for row in results:
            date, runs, acc, lat, fp, fn = row
            print(f"{date:<12} {runs:<6} {acc:.1%}      {lat:.0f}ms     {fp or 0:<5} {fn or 0:<5}")
    
    # Parameter changes
    print(f"\n🔧 Recent Parameter Changes")
    print("=" * 50)
    
    params = conn.execute('''
        SELECT timestamp, min_lip_movement, min_mar_for_speech, 
               asd_threshold, smoothing_window, reason
        FROM parameter_changes
        WHERE timestamp > datetime('now', ?)
        ORDER BY timestamp DESC
        LIMIT 10
    ''', (f'-{days} days',)).fetchall()
    
    if not params:
        print("No parameter changes in this period.")
    else:
        for row in params:
            ts, lip, mar, thresh, window, reason = row
            print(f"\n{ts[:16]}")
            if lip: print(f"  min_lip_movement: {lip}")
            if mar: print(f"  min_mar_for_speech: {mar}")
            if thresh: print(f"  asd_threshold: {thresh}")
            if window: print(f"  smoothing_window: {window}")
            if reason: print(f"  → {reason}")


def export_data(output_path="metrics_export.json"):
    """Export all data to JSON"""
    conn = get_connection()
    
    data = {
        'exported_at': datetime.now().isoformat(),
        'test_runs': [],
        'parameter_changes': [],
    }
    
    # Export test runs
    for row in conn.execute('SELECT * FROM test_runs ORDER BY timestamp'):
        data['test_runs'].append({
            'id': row[0],
            'timestamp': row[1],
            'accuracy': row[2],
            'latency_ms': row[3],
            'false_positives': row[4],
            'false_negatives': row[5],
            'environment': row[6],
            'config': json.loads(row[7]) if row[7] else None,
            'notes': row[8]
        })
    
    # Export parameter changes
    for row in conn.execute('SELECT * FROM parameter_changes ORDER BY timestamp'):
        data['parameter_changes'].append({
            'id': row[0],
            'timestamp': row[1],
            'min_lip_movement': row[2],
            'min_mar_for_speech': row[3],
            'asd_threshold': row[4],
            'smoothing_window': row[5],
            'reason': row[6]
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Exported to {output_path}")
    print(f"   {len(data['test_runs'])} test runs")
    print(f"   {len(data['parameter_changes'])} parameter changes")


def get_current_config():
    """Get current configuration (placeholder - integrate with your config)"""
    try:
        # Try to import from your config module
        from smait.core.config import get_config
        c = get_config()
        return {
            'min_lip_movement': c.min_lip_movement,
            'min_mar_for_speech': c.min_mar_for_speech,
            'asd_threshold': c.asd_threshold,
            'smoothing_window': c.smoothing_window,
        }
    except:
        # Fallback to reading config.yaml
        try:
            import yaml
            with open('config.yaml') as f:
                return yaml.safe_load(f)
        except:
            return {'error': 'Could not read config'}


def main():
    parser = argparse.ArgumentParser(description='SMAIT Metrics Tracker')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Log test run
    log_parser = subparsers.add_parser('log', help='Log a test run')
    log_parser.add_argument('--accuracy', '-a', type=float, required=True)
    log_parser.add_argument('--latency', '-l', type=float, required=True)
    log_parser.add_argument('--fp', type=int, default=0, help='False positives')
    log_parser.add_argument('--fn', type=int, default=0, help='False negatives')
    log_parser.add_argument('--env', '-e', default='unknown', help='Environment')
    log_parser.add_argument('--notes', '-n', default='', help='Notes')
    
    # Log parameter change
    params_parser = subparsers.add_parser('params', help='Log parameter change')
    params_parser.add_argument('--lip', type=float, help='min_lip_movement')
    params_parser.add_argument('--mar', type=float, help='min_mar_for_speech')
    params_parser.add_argument('--threshold', '-t', type=float, help='asd_threshold')
    params_parser.add_argument('--window', '-w', type=int, help='smoothing_window')
    params_parser.add_argument('--reason', '-r', default='', help='Reason for change')
    
    # Show trends
    trends_parser = subparsers.add_parser('trends', help='Show trends')
    trends_parser.add_argument('--days', '-d', type=int, default=7)
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('--output', '-o', default='metrics_export.json')
    
    args = parser.parse_args()
    
    if args.command == 'log':
        log_test_run(args.accuracy, args.latency, args.fp, args.fn, args.env, args.notes)
    elif args.command == 'params':
        log_parameter_change(args.lip, args.mar, args.threshold, args.window, args.reason)
    elif args.command == 'trends':
        show_trends(args.days)
    elif args.command == 'export':
        export_data(args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
