from pathlib import Path
import pandas as pd
from tools.summarize_v3 import summarize_one

def test_summarize_one_smoke(tmp_path: Path):
    d = tmp_path / "rep"; d.mkdir()
    # 3 fichiers minimaux
    pd.DataFrame({"date":["2024-01-01","2024-01-02"],"equity":[1.0,1.01]}).to_csv(d/"equity_curve.csv", index=False)
    pd.DataFrame([{"date":"2024-01-02","rnet":0.01,"rgross":0.011,"cost":0.001}]).to_csv(d/"daily_net_vs_gross.csv", index=False)
    pd.DataFrame([{"date":"2024-01-02","turnover":1.0,"cost":0.001}]).to_csv(d/"trades_daily.csv", index=False)

    row = summarize_one(d)
    assert "ann_return_%".replace("%","_%") or "ann_return_%"  # juste pour importer la fonction
    assert row["days"] == 2
    assert "sharpe" in row
