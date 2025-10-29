import pandas as pd
from pathlib import Path
from oro.engine.run_backtest_v3 import run_backtest_v3

def test_run_backtest_v3_smoke(tmp_path: Path):
    # config minimale
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "start: 2024-03-28\n"
        "end:   2024-04-12\n"
        "seed:  123\n"
        "equity_start: 1.0\n"
        "universe_size: 10\n"
        "costs: { bps: 5.0 }\n",
        encoding="utf-8"
    )
    out_dir = tmp_path / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # run
    paths = run_backtest_v3(cfg_path, out_dir)

    # fichiers attendus
    for fn in [
        "equity_curve.csv","trades.csv",
        "trades_daily.csv","daily_net_vs_gross.csv",
        "metrics.yaml","report.yaml",
    ]:
        assert (out_dir / fn).exists(), f"manquant: {fn}"

    # format equity
    ec = pd.read_csv(out_dir / "equity_curve.csv")
    assert {"date","equity"}.issubset(ec.columns)
    assert len(ec) >= 2

    # format trades
    tr = pd.read_csv(out_dir / "trades.csv")
    for c in ["date","ticker","w_prev","w_new","turnover_piece"]:
        assert c in tr.columns

    # cohérence net vs gross
    dng = pd.read_csv(out_dir / "daily_net_vs_gross.csv")
    if len(dng):
        assert ((dng["rnet"] + dng["cost"] - dng["rgross"]).abs() < 1e-9).all()
