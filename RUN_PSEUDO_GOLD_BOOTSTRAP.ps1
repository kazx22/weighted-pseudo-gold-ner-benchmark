Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path "src\bootstrap_significance.py")) {
    Write-Host "ERROR: src\bootstrap_significance.py was not found."
    exit 1
}

$RequiredFiles = @(
    "data\processed\bc5cdr\bc5cdr_test_docs.jsonl",
    "data\processed\bc5cdr\bc5cdr_test_entities.jsonl",
    "data\processed\bc5cdr\scispacy_test_entities_bc5cdr.jsonl",
    "data\gold\majority_pseudo_gold_test_entities_bc5cdr.jsonl",
    "data\gold\weighted_pseudo_gold_test_entities_bc5cdr.jsonl"
)

foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        Write-Host "ERROR: Required existing result is missing: $File"
        Write-Host "Do not rerun models yet. Check that you are in the correct project root."
        exit 1
    }
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    Write-Host "ERROR: Python was not found. Activate .venv first."
    exit 1
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $PSScriptRoot "logs\pseudo_gold_bootstrap_$RunStamp.log"

Set-Content -LiteralPath $LogFile -Value "BC5CDR PSEUDO-GOLD BOOTSTRAP ONLY" -Encoding UTF8
Add-Content -LiteralPath $LogFile -Value "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
Add-Content -LiteralPath $LogFile -Value "Project: $PSScriptRoot" -Encoding UTF8
Add-Content -LiteralPath $LogFile -Value "Python: $($PythonCommand.Source)" -Encoding UTF8

Write-Host "Running bootstrap only. No models will be rerun."
Write-Host "Log: $LogFile"
Write-Host ""

& python -m src.bootstrap_significance --split test --resamples 1000 2>&1 |
    Tee-Object -FilePath $LogFile -Append |
    ForEach-Object { Write-Host $_ }

$ExitCode = $LASTEXITCODE
if ($null -eq $ExitCode) {
    $ExitCode = 0
}

if ($ExitCode -ne 0) {
    Write-Host ""
    Write-Host "BOOTSTRAP FAILED with exit code $ExitCode."
    Write-Host "Check: $LogFile"
    exit $ExitCode
}

Add-Content -LiteralPath $LogFile -Value "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8

Write-Host ""
Write-Host "BOOTSTRAP COMPLETED SUCCESSFULLY."
Write-Host "Main new result: results\test\bootstrap_primary_comparisons.csv"
Write-Host "Updated JSON: results\test\bootstrap_results.json"
Write-Host "Log: $LogFile"
exit 0
