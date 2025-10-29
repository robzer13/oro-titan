# tools/run_tests.ps1
$ErrorActionPreference = "Stop"
$Root = Join-Path $PSScriptRoot ".." | Resolve-Path
$env:PYTHONPATH = (Join-Path $Root "src")  # oro.*
pytest -q
