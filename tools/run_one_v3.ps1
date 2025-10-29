param(
  [Parameter(Mandatory=$true)] [string]$Config,      # ex: .\configs\best_v3_oos_15d.yaml
  [Parameter(Mandatory=$true)] [string]$ReportDir    # ex: .\reports\v3_custom_15d
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSCommandPath
Set-Location $root\..

$env:PYTHONPATH = "$PWD\src"

python -m oro.engine.run_backtest_v3 --config $Config --report-dir $ReportDir

# Résumé ciblé (table + _index_v3_plus.csv mis à jour)
powershell -File .\tools\summarize.ps1 $ReportDir
