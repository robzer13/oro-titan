#!/usr/bin/env python3
"""
Smoke test for OroTitan data -> factors -> backtest pipeline.

Usage:
    python scripts/smoke_pipeline.py

This script verifies that the full pipeline works end-to-end using synthetic data.
It checks file existence, generates synthetic prices, adds factors, runs backtest,
and validates output CSVs.
"""
import os
import sys
import subprocess
import time
import csv
import datetime
from pathlib import Path

# Scripts root directory (parent of scripts/)
SCRIPT_ROOT = Path(__file__).parent.parent


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    return filepath.exists() and filepath.is_file()


def check_files() -> bool:
    """Check existence of critical files."""
    files = [
        SCRIPT_ROOT / "scripts" / "validate_prices.py",
        SCRIPT_ROOT / "scripts" / "fix_prices.py",
        SCRIPT_ROOT / "scripts" / "build_prices_yf.py",
        SCRIPT_ROOT / "scripts" / "make_synth_prices.py",
        SCRIPT_ROOT / "scripts" / "add_factors.py",
        SCRIPT_ROOT / "backtest_runner.py",
    ]
    
    missing = []
    for f in files:
        if not check_file_exists(f):
            missing.append(str(f))
    
    if missing:
        print("[smoke] ERROR: missing required files:")
        for m in missing:
            print(f"  - {m}")
        return False
    
    return True


def run_command(cmd: list, description: str) -> bool:
    """Run a command and check its exit code."""
    print(f"[smoke] {description}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"[smoke] ERROR: command failed with exit code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[smoke] ERROR: exception running command: {e}")
        return False


def check_output_files() -> bool:
    """Check that output CSVs exist and are non-empty."""
    outputs = [
        SCRIPT_ROOT / "out" / "curve_smoke.csv",
        SCRIPT_ROOT / "out" / "weights_log_smoke.csv",
        SCRIPT_ROOT / "out" / "picks_log_smoke.csv",
    ]
    
    missing_or_empty = []
    for f in outputs:
        if not check_file_exists(f):
            missing_or_empty.append(f"missing: {f}")
        elif f.stat().st_size == 0:
            missing_or_empty.append(f"empty: {f}")
    
    if missing_or_empty:
        print("[smoke] ERROR: backtest outputs missing or empty:")
        for msg in missing_or_empty:
            print(f"  - {msg}")
        return False
    
    return True


def get_factor_rows() -> int:
    """Get row count from factors parquet if available."""
    factors_file = SCRIPT_ROOT / "prices_with_factors_smoke.parquet"
    if factors_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(factors_file)
            return len(df)
        except Exception:
            return 0
    return 0


def log_pipeline_check(status: str, duration_sec: float, rows: int, comment: str):
    """Log pipeline check result to CSV."""
    logs_dir = SCRIPT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "pipeline_checks.csv"
    
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    file_exists = log_file.exists()
    with open(log_file, "a", newline="", encoding="ascii") as f:
        writer = csv.writer(f, delimiter=",")
        if not file_exists:
            writer.writerow(["date", "time", "type", "status", "duration_sec", "rows", "comment"])
        writer.writerow([date_str, time_str, "synthetic", status, f"{duration_sec:.1f}", rows, comment])


def main() -> int:
    """Run the smoke test pipeline."""
    start_time = time.time()
    status = "FAIL"
    comment = "-"
    rows = 0
    
    try:
        # 1. Check file existence
        print("[smoke] Checking required files...")
        if not check_files():
            comment = "missing files"
            return 2
        
        # 2. Generate synthetic data
        synth_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "scripts" / "make_synth_prices.py"),
            str(SCRIPT_ROOT / "prices_synth_smoke.parquet"),
        ]
        if not run_command(synth_cmd, "Generating synthetic prices"):
            comment = "synth failed"
            return 2
        
        # 3. Add factors
        factors_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "scripts" / "add_factors.py"),
            str(SCRIPT_ROOT / "prices_synth_smoke.parquet"),
            str(SCRIPT_ROOT / "prices_with_factors_smoke.parquet"),
        ]
        if not run_command(factors_cmd, "Adding factors"):
            comment = "factors failed"
            return 2
        
        # 4. Create out directory if needed
        out_dir = SCRIPT_ROOT / "out"
        out_dir.mkdir(exist_ok=True)
        
        # 5. Run backtest
        backtest_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "backtest_runner.py"),
            "--prices", str(SCRIPT_ROOT / "prices_with_factors_smoke.parquet"),
            "--config", str(SCRIPT_ROOT / "configs" / "scoring.yml"),
            "--date_start", "2018-01-01",
            "--curve_out", str(SCRIPT_ROOT / "out" / "curve_smoke.csv"),
            "--weights_out", str(SCRIPT_ROOT / "out" / "weights_log_smoke.csv"),
            "--picks_out", str(SCRIPT_ROOT / "out" / "picks_log_smoke.csv"),
        ]
        if not run_command(backtest_cmd, "Running backtest"):
            comment = "backtest failed"
            return 2
        
        # 6. Check output files
        print("[smoke] Checking output files...")
        if not check_output_files():
            comment = "outputs missing or empty"
            return 2
        
        # Get row count from factors
        rows = get_factor_rows()
        
        # Success
        status = "OK"
        comment = "-"
        print("[smoke] PIPELINE OK")
        return 0
    finally:
        duration = time.time() - start_time
        log_pipeline_check(status, duration, rows, comment)


if __name__ == "__main__":
    sys.exit(main())

