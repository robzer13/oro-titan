import pandas as pd
from pathlib import Path
from oro.engine.run_backtest_v3 import run_backtest_v3

def _write_cfg(p: Path):
    p.write_text(
        "start: 2024-03-28\n"
        "end:   2024-04-12\n"
        "seed:  123\n"
        "equity_start: 1.0\n"
        "universe_size: 10\n"
        "costs: { bps: 5.0 }\n",
        encoding="utf-8"
    )

def test_v3_determinism_same_seed(tmp_path: Path):
    cfg1 = tmp_path / "cfg1.yaml"; _write_cfg(cfg1)
    cfg2 = tmp_path / "cfg2.yaml"; _write_cfg(cfg2)
    out1 = tmp_path / "out1"; out2 = tmp_path / "out2"
    out1.mkdir(); out2.mkdir()

    run_backtest_v3(cfg1, out1)
    run_backtest_v3(cfg2, out2)

    e1 = pd.read_csv(out1 / "equity_curve.csv")
    e2 = pd.read_csv(out2 / "equity_curve.csv")
    assert len(e1)==len(e2)
    assert (e1["equity"].round(12).equals(e2["equity"].round(12)))
