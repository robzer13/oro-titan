#!/usr/bin/env python3
import argparse, itertools, json, subprocess, sys, tempfile, copy, os, yaml

def run_backtest(prices, config_path, top_k, rebalance, fees_bps, min_score_z):
    cmd = [sys.executable, "backtest_runner.py",
           "--prices", prices, "--config", config_path,
           "--top_k", str(top_k), "--rebalance", rebalance,
           "--fees_bps", str(fees_bps), "--min_score_z", str(min_score_z)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    txt = out.stdout
    # parse hyper simple
    def get(metric):
        for line in txt.splitlines():
            if line.strip().startswith(metric+":"):
                return float(line.split(":")[1].replace("%","").strip())
        return None
    return {
        "TotalReturn_%": get("TotalReturn"),
        "CAGR_%": get("CAGR"),
        "Vol_%": get("Vol"),
        "Sharpe": get("Sharpe"),
        "Sortino": get("Sortino"),
        "MaxDD_%": get("MaxDD"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="prices.parquet")
    ap.add_argument("--base_config", default="scoring_config.yml")
    ap.add_argument("--out", default="tuning_results.json")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--rebalance", default="W-FRI")
    ap.add_argument("--fees_bps", type=float, default=5)
    ap.add_argument("--min_score_z", type=float, default=0.0)
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base_config, "r", encoding="utf-8"))
    base_weights = base.get("weights", {})

    # Définis des multiplicateurs par “famille” de signaux
    grids = {
        "trend":   {"macd_diff_z":[0.8,1.0,1.2], "mom_12_1_z":[0.8,1.0,1.2]},
        "meanrev": {"rev_5_z":[-0.2,-0.4,-0.6], "bbp_z":[0.4,0.6,0.8]},
        "risk":    {"atr14_n_z":[-0.1,-0.2,-0.4], "rv_30_z":[-0.1,-0.2,-0.3], "vol_z20_z":[-0.1,-0.2,-0.3]},
        "breakout":{"donchian_break20_z":[0.7,0.9,1.1]},
        "rsi":     {"rsi_14_z":[0.6,0.8,1.0]},
    }

    keys, values = [], []
    for block in grids.values():
        for k, arr in block.items():
            if k in base_weights:       # ne gridsearch que sur ce qui existe
                keys.append(k); values.append(arr)

    results = []
    for combo in itertools.product(*values):
        w = copy.deepcopy(base_weights)
        for k, mult in zip(keys, combo):
            w[k] = round(mult, 3)
        tmp_cfg = copy.deepcopy(base)
        tmp_cfg["weights"] = w
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(tmp_cfg, f, sort_keys=False, allow_unicode=True)
            tmp_path = f.name
        try:
            metrics = run_backtest(args.prices, tmp_path, args.top_k, args.rebalance, args.fees_bps, args.min_score_z)
            row = {"weights": w}
            row.update(metrics or {})
            results.append(row)
            print("→", row)
        finally:
            os.remove(tmp_path)

    # tri multi-critères: Sharpe haut, MaxDD bas, CAGR haut
    def score(row):
        s = (row.get("Sharpe") or 0.0) - 0.01*(row.get("MaxDD_%") or 0.0) + 0.002*(row.get("CAGR_%") or 0.0)
        return s
    results.sort(key=score, reverse=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nTop 5 configs:")
    for r in results[:5]:
        print(r)

if __name__ == "__main__":
    main()
