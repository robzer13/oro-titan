import os
from pathlib import Path
import pytest

from oro.engine.run_backtest_v3 import run_backtest_v3

CFG_LIST = [
    Path("configs/best_v3_oos_10d.yaml"),
    Path("configs/best_v3_oos_15d.yaml"),
    Path("configs/best_v3_oos_20d.yaml"),
    Path("configs/best_v3_oos_30d.yaml"),
]

@pytest.mark.parametrize("cfg_path", CFG_LIST)
def test_v3_configs_exist_and_run(tmp_path: Path, cfg_path: Path):
    if not cfg_path.exists():
        pytest.skip(f"config absente: {cfg_path}")
    out = tmp_path / "rep"; out.mkdir()
    run_backtest_v3(cfg_path, out)
    for fn in ["equity_curve.csv","trades_daily.csv","daily_net_vs_gross.csv","metrics.yaml","report.yaml"]:
        assert (out / fn).exists(), f"manquant: {fn}"
