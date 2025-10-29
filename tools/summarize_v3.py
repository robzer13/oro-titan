import sys
from pathlib import Path
import math
import pandas as pd
import numpy as np

def ann_from_cum(mult, n_days, base=252):
    if n_days <= 0 or mult <= 0:
        return float("nan")
    return mult**(base/n_days) - 1.0

def max_drawdown_from_equity(ec: pd.DataFrame) -> float:
    x = ec["equity"].astype(float).values
    peak = -1e30
    mdd = 0.0
    for v in x:
        peak = max(peak, v)
        mdd = min(mdd, v/peak - 1.0)
    return float(mdd)

def summarize_one(report_dir: Path) -> dict:
    ec = pd.read_csv(report_dir / "equity_curve.csv")
    dng = pd.read_csv(report_dir / "daily_net_vs_gross.csv")
    daily = pd.read_csv(report_dir / "trades_daily.csv")

    days = len(ec)
    first = float(ec["equity"].iloc[0])
    last  = float(ec["equity"].iloc[-1])
    mult  = (last/first) if first else float("nan")

    rnet = dng["rnet"].astype(float) if len(dng) else pd.Series([], dtype=float)
    wins = int((rnet > 0).sum())
    loss = int((rnet < 0).sum())
    tot  = wins + loss
    winp = (wins/tot) if tot else float("nan")

    ann_ret = ann_from_cum(mult, days)
    ann_vol = (rnet.std(ddof=0) * math.sqrt(252)) if len(rnet) else float("nan")
    sharpe  = (ann_ret/ann_vol) if (ann_vol and ann_vol==ann_vol and ann_vol>0) else float("nan")

    mdd = max_drawdown_from_equity(ec)

    turn = float(daily["turnover"].sum()) if "turnover" in daily.columns else float("nan")
    cost = float(daily["cost"].sum()) if "cost" in daily.columns else float("nan")

    return {
        "window": report_dir.name,
        "start": ec["date"].iloc[0],
        "end": ec["date"].iloc[-1],
        "days": days,
        "cumret_%": 100*(mult-1),
        "ann_return_%": 100*ann_ret,
        "ann_vol_%": 100*ann_vol if ann_vol==ann_vol else float("nan"),
        "sharpe": sharpe,
        "win_%": 100*winp if winp==winp else float("nan"),
        "maxDD_%": 100*mdd,
        "turnover": turn,
        "cost_%": 100*cost if cost==cost else float("nan"),
    }

def main(args):
    if not args:
        base = Path("reports")
        paths = sorted([p for p in base.glob("v3_*") if p.is_dir()])
    else:
        paths = [Path(a) for a in args]

    rows = []
    for p in paths:
        need = ["equity_curve.csv","daily_net_vs_gross.csv","trades_daily.csv"]
        if not all((p/f).exists() for f in need):
            print(f"[WARN] Incomplet: {p}")
            continue
        rows.append(summarize_one(p))
    if not rows:
        print("Rien à résumer.")
        return

    df = pd.DataFrame(rows)
    out = Path("reports") / "_index_v3_plus.csv"
    df.to_csv(out, index=False)

    show = df.copy()
    for c in ["cumret_%","ann_return_%","ann_vol_%","win_%","maxDD_%","cost_%"]:
        show[c] = show[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    show["sharpe"] = show["sharpe"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    print(show.to_string(index=False))
    print(f"\nÉcrit -> {out.resolve()}")

if __name__ == "__main__":
    main(sys.argv[1:])
