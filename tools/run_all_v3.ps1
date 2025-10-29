# tools/run_all_v3.ps1
param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$Root = Join-Path $PSScriptRoot ".." | Resolve-Path

# 1) Env pour importer oro.*
$env:PYTHONPATH = (Join-Path $Root "src")

# 2) Jobs
$jobs = @(
  @{ cfg = (Join-Path $Root "configs/best_v3_oos_10d.yaml"); dir = (Join-Path $Root "reports/v3_oos_10d") },
  @{ cfg = (Join-Path $Root "configs/best_v3_oos_15d.yaml"); dir = (Join-Path $Root "reports/v3_oos_15d") },
  @{ cfg = (Join-Path $Root "configs/best_v3_oos_20d.yaml"); dir = (Join-Path $Root "reports/v3_oos_20d") },
  @{ cfg = (Join-Path $Root "configs/best_v3_oos_30d.yaml"); dir = (Join-Path $Root "reports/v3_oos_30d") }
)

foreach($j in $jobs){
  & $PythonExe -m oro.engine.run_backtest_v3 --config $j.cfg --report-dir $j.dir
}

# 3) Résumé agrégé
$SumPy = Join-Path $PSScriptRoot "summarize_v3.py"
& $PythonExe $SumPy
