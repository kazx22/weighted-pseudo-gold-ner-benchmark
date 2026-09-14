Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
} catch {
}
$env:PYTHONIOENCODING = "utf-8"

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    Write-Host "ERROR: Python was not found. Activate .venv first."
    exit 1
}

$RequiredFiles = @(
    "results\dev\frozen_pseudo_gold_config.json",
    "data\processed\bc5cdr\bc5cdr_dev_docs.jsonl",
    "data\processed\bc5cdr\bc5cdr_test_docs.jsonl",
    "data\processed\bc5cdr\scispacy_dev_entities_bc5cdr.jsonl",
    "data\processed\bc5cdr\scispacy_test_entities_bc5cdr.jsonl"
)
foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        Write-Host "ERROR: Required baseline file is missing: $File"
        Write-Host "Run the original development and test pipeline first."
        exit 1
    }
}

try {
    $Version = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method Get -TimeoutSec 10
    Write-Host "Ollama version: $($Version.version)"
} catch {
    Write-Host "ERROR: Ollama is not running at http://localhost:11434"
    Write-Host "Start Ollama, then run this file again."
    exit 1
}

$ModelName = "medgemma1.5:4b-it-q4_K_M"
$Tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 30
$Installed = $false
foreach ($Model in $Tags.models) {
    if ($Model.name -eq $ModelName -or $Model.model -eq $ModelName) {
        $Installed = $true
    }
}
if (-not $Installed) {
    Write-Host "ERROR: $ModelName is not installed."
    Write-Host "Run: ollama pull $ModelName"
    exit 1
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $PSScriptRoot "logs\medgemma_bc5cdr_$Stamp.log"

$Commands = @(
    @("-m", "src.medgemma_smoke_test"),
    @("-m", "src.medgemma_bc5cdr", "--split", "dev"),
    @("-m", "src.medgemma_hybrid", "--split", "dev"),
    @("-m", "src.medgemma_evaluation", "--split", "dev"),
    @("-m", "src.medgemma_bc5cdr", "--split", "test"),
    @("-m", "src.medgemma_hybrid", "--split", "test"),
    @("-m", "src.medgemma_evaluation", "--split", "test"),
    @("-m", "src.medgemma_bootstrap", "--split", "test", "--resamples", "1000")
)

function Write-Log {
    param([string]$Text = "")
    Write-Host $Text
    Add-Content -LiteralPath $script:LogFile -Value $Text -Encoding UTF8
}

Set-Content -LiteralPath $LogFile -Value "MEDGEMMA BC5CDR PIPELINE" -Encoding UTF8
Write-Log "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Model: $ModelName"
Write-Log "This extension uses existing five-model predictions and does not rerun them."

foreach ($Arguments in $Commands) {
    $Display = "python " + ($Arguments -join " ")
    Write-Log ""
    Write-Log "============================================================"
    Write-Log "RUNNING: $Display"
    Write-Log "============================================================"
    # Windows PowerShell 5.1 can turn native stderr into a terminating
    # NativeCommandError when ErrorActionPreference is Stop. Temporarily use
    # Continue so the complete Python traceback is printed and logged.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python @Arguments 2>&1 | ForEach-Object {
        $Line = $_.ToString()
        Write-Host $Line
        Add-Content -LiteralPath $script:LogFile -Value $Line -Encoding UTF8
    }
    $PythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    if ($PythonExitCode -ne 0) {
        Write-Log "FAILED: $Display"
        Write-Log "Resume by running 05_RUN_MEDGEMMA_BC5CDR.cmd again."
        exit $PythonExitCode
    }
}

Write-Log ""
Write-Log "MEDGEMMA BC5CDR PIPELINE COMPLETED SUCCESSFULLY."
Write-Log "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Main results: results\test\medgemma_evaluation.json"
Write-Log "Bootstrap: results\test\medgemma_bootstrap_results.json"
Write-Host "Full log: $LogFile"
exit 0
