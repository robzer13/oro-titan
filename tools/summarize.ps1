# tools/summarize.ps1
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Reports
)

$ErrorActionPreference = "Stop"
$Py = "python"
$Script = Join-Path $PSScriptRoot "summarize_v3.py"

if($Reports -and $Reports.Count -gt 0){
  & $Py $Script @Reports
} else {
  & $Py $Script   # scan reports/v3_* automatiquement
}
