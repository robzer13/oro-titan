# tools/smoke_v2.py
import sys, json
from pathlib import Path
import subprocess as sp
import csv, yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

def run(cmd):
    print("[RUN]", " ".join(cmd))
    cp = sp.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(cp.stdout)
    if cp.returncode != 0:
        print(cp.stderr)
        raise SystemExit(cp.returncode)

def read_csv_head(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        return next(r)

def parse_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    for k in ("cagr","vol","sharpe","max_drawdown"):
        assert k in y, f"metrics.yaml: '{k}' manquant"
    # check types
    float(y["cagr"]); float(y["vol"]); float(y["sharpe"]); float(y["max_drawdown"])

def main():
    outdir = ROOT/"reports/smoke_v2"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) sweep rapide
    run([PY,"-m","cli.tune_v2","--base-config","configs/config.yaml",
         "--sweep","configs/sweep_v2.yaml","--outdir", str(outdir)])

    # 2) IS
    run([PY,"-m","cli.backtest_v2","--config", str(outdir/"best_config.yaml"),
         "--report-dir", str(outdir/"IS")])

    # 3) OOS (15j)
    run([PY,"tools/make_oos_from_best.py","--best", str(outdir/"best_config.yaml"),
         "--prices","data_proc/prices_eod.csv","--oos-days","15",
         "--out", str(outdir/"best_oos.yaml")])
    run([PY,"-m","cli.backtest_v2","--config", str(outdir/"best_oos.yaml"),
         "--report-dir", str(outdir/"OOS")])

    # 4) validations
    for rep in ("IS","OOS"):
        ec = outdir/rep/"equity_curve.csv"
        hdr = read_csv_head(ec)
        assert hdr == ["date","equity"], f"{ec} entête invalide: {hdr}"
        parse_metrics(outdir/rep/"metrics.yaml")

    print("\n✅ SMOKE PASS: v2 ok (I/O + métriques).")

if __name__ == "__main__":
    main()
