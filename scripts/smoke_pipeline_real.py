#!/usr/bin/env python3
"""
Smoke test for OroTitan data -> factors -> backtest pipeline using REAL data from yfinance.

Usage:
    python scripts/smoke_pipeline_real.py

This script verifies that the full pipeline works end-to-end using real data from
yfinance via build_prices_yf.py. It checks file existence, builds prices from
tickers.txt, validates the parquet, adds factors, runs backtest, and validates output CSVs.
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
        SCRIPT_ROOT / "scripts" / "add_factors.py",
        SCRIPT_ROOT / "backtest_runner.py",
        SCRIPT_ROOT / "tickers.txt",
    ]
    
    missing = []
    for f in files:
        if not check_file_exists(f):
            missing.append(str(f))
    
    if missing:
        for m in missing:
            print(f"[smoke-real] MISSING: {m}")
        return False
    
    return True


def run_command(cmd: list, description: str, check_stderr: bool = False) -> tuple[bool, int]:
    """Run a command and return (success, exit_code)."""
    print(f"[smoke-real] {description}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"[smoke-real] ERROR: command failed with exit code {result.returncode}")
            if result.stderr and check_stderr:
                print(f"  stderr: {result.stderr[:200]}")
            return False, result.returncode
        return True, 0
    except Exception as e:
        print(f"[smoke-real] ERROR: exception running command: {e}")
        return False, 1


def check_output_files() -> bool:
    """Check that output CSVs exist and are non-empty."""
    outputs = [
        SCRIPT_ROOT / "out" / "curve_real.csv",
        SCRIPT_ROOT / "out" / "weights_log_real.csv",
        SCRIPT_ROOT / "out" / "picks_log_real.csv",
    ]
    
    missing_or_empty = []
    for f in outputs:
        if not check_file_exists(f):
            missing_or_empty.append(f"missing: {f}")
        elif f.stat().st_size == 0:
            missing_or_empty.append(f"empty: {f}")
    
    if missing_or_empty:
        print("[smoke-real] ERROR: backtest outputs missing or empty:")
        for msg in missing_or_empty:
            print(f"  - {msg}")
        return False
    
    return True


def get_factor_rows() -> int:
    """Get row count from factors parquet if available."""
    factors_file = SCRIPT_ROOT / "prices_with_factors_real.parquet"
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
        writer.writerow([date_str, time_str, "real", status, f"{duration_sec:.1f}", rows, comment])


def main() -> int:
    """Run the smoke test pipeline with real data."""
    start_time = time.time()
    status = "FAIL"
    comment = "-"
    rows = 0
    
    try:
        # 1. Check file existence
        print("[smoke-real] Checking required files...")
        if not check_files():
            comment = "missing files"
            return 2
    
        # 2. Build prices from yfinance
        build_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "scripts" / "build_prices_yf.py"),
            str(SCRIPT_ROOT / "tickers.txt"),
            str(SCRIPT_ROOT / "prices_built_real.parquet"),
            "2015-01-01",
        ]
        success, exit_code = run_command(build_cmd, "Building prices from yfinance")
        if not success:
            comment = "build yfinance failed"
            print("[smoke-real] ERROR: build yfinance failed")
            return 2
        
        # 3. Validate the built parquet
        validate_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "scripts" / "validate_prices.py"),
            str(SCRIPT_ROOT / "prices_built_real.parquet"),
        ]
        success, exit_code = run_command(validate_cmd, "Validating prices")
        if not success:
            if exit_code == 2:
                comment = "no usable close"
                print("[smoke-real] ERROR: built prices have no usable close")
            else:
                comment = "validation failed"
            return 2
        
        # 4. Add factors
        factors_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "scripts" / "add_factors.py"),
            str(SCRIPT_ROOT / "prices_built_real.parquet"),
            str(SCRIPT_ROOT / "prices_with_factors_real.parquet"),
        ]
        success, exit_code = run_command(factors_cmd, "Adding factors")
        if not success:
            comment = "factors failed"
            return 2
        
        # 5. Create out directory if needed
        out_dir = SCRIPT_ROOT / "out"
        out_dir.mkdir(exist_ok=True)
        
        # 6. Run backtest
        backtest_cmd = [
            sys.executable,
            str(SCRIPT_ROOT / "backtest_runner.py"),
            "--prices", str(SCRIPT_ROOT / "prices_with_factors_real.parquet"),
            "--config", str(SCRIPT_ROOT / "configs" / "scoring.yml"),
            "--date_start", "2018-01-01",
            "--curve_out", str(SCRIPT_ROOT / "out" / "curve_real.csv"),
            "--weights_out", str(SCRIPT_ROOT / "out" / "weights_log_real.csv"),
            "--picks_out", str(SCRIPT_ROOT / "out" / "picks_log_real.csv"),
        ]
        success, exit_code = run_command(backtest_cmd, "Running backtest")
        if not success:
            comment = "backtest failed"
            return 2
        
        # 7. Check output files
        print("[smoke-real] Checking output files...")
        if not check_output_files():
            comment = "outputs missing or empty"
            return 2
        
        # Get row count from factors
        rows = get_factor_rows()

        # Check prices profile for non-OK tickers
        profile_comment = "build OK"
        profile_file = SCRIPT_ROOT / "out" / "prices_profile.csv"
        if profile_file.exists():
            try:
                with open(profile_file, newline="", encoding="ascii") as profile_csv:
                    reader = csv.DictReader(profile_csv)
                    not_ok_count = 0
                    for row in reader:
                        status_value = (row.get("status", "") or "").strip()
                        if status_value.upper() != "OK":
                            not_ok_count += 1
                    if not_ok_count > 0:
                        profile_comment = f"profile: {not_ok_count} tickers not OK"
            except Exception as exc:
                print(f"[smoke-real] WARNING: unable to read prices_profile.csv: {exc}")
                profile_comment = "profile read error"

        # Success
        status = "OK"
        comment = profile_comment
        print("[smoke-real] PIPELINE REAL OK")
        return 0
    finally:
        duration = time.time() - start_time
        log_pipeline_check(status, duration, rows, comment)


if __name__ == "__main__":
    sys.exit(main())

