#!/usr/bin/env python3
import os, subprocess, sys, yaml

with open("config_prices.yml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cmd = [
    sys.executable, "prices_downloader.py",
    "--universe", cfg["universe_csv"],
    "--start", cfg["start"],
    "--end", cfg["end"],
    "--backend", cfg["backend"],
    "--out", cfg["out_file"],
    "--format", cfg["format"],
    "--benchmarks", cfg.get("benchmarks",""),
    "--min_rows", str(cfg.get("min_rows", 100))
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Done.")

export RAPIDAPI_KEY= "ed38eacaa0mshfa79856eeaf7605p14f6cejsn7f182f522b89"
