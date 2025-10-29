import pandas as pd
from pathlib import Path
from oro.engine.run_backtest_v3 import run_backtest_v3
import numpy as np

def test_v3_positions_and_signals_shape(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "start: 2024-03-28\n"
        "end:   2024-04-12\n"
        "seed:  123\n"
        "equity_start: 1.0\n"
        "universe_size: 10\n"
        "costs: { bps: 5.0 }\n",
        encoding="utf-8"
    )
    out = tmp_path / "out"; out.mkdir()
    run_backtest_v3(cfg, out)

    pos_p = out / "positions.csv"
    sig_p = out / "signals.csv"
    if pos_p.exists() and sig_p.exists():
        pos = pd.read_csv(pos_p)
        sig = pd.read_csv(sig_p)
        pc = sorted([c for c in pos.columns if c!="date"])
        sc = sorted([c for c in sig.columns if c!="date"])
        assert pc == sc, "colonnes positions vs signals"
        # somme des poids ≈ 1
        s = pos[pc].sum(axis=1)
        assert np.allclose(s.values, 1.0, atol=1e-2)
