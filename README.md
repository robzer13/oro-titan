# OroTitan V2 – Backtest runner & artéfacts

## Lancer
```powershell
$env:PYTHONPATH = "$PWD\src"
python -m oro.engine.run_backtest_v2 --config .\configs\best_v2_oos_15d.yaml --report-dir .\reports\fact_best_oos_15d

# OroTitan – Backtest v3 (Quick Start)

## 1) Lancer tous les backtests
```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_all_v3.ps1
```

## Data Layer

Pour le pipeline data (validate -> fix -> build -> factors -> backtest), voir [DOCS\data_layer.md](DOCS/data_layer.md).
