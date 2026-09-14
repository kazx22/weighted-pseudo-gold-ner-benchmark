param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("development", "test")]
    [string]$Stage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path "src\experiment_config.py")) {
    Write-Host "ERROR: src\experiment_config.py was not found."
    Write-Host "Place this file and the CMD files in the project root beside src and data."
    exit 1
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    Write-Host "ERROR: Python was not found in this CMD session."
    Write-Host "Activate your virtual environment first, then run the CMD file again."
    exit 1
}

if ($Stage -eq "test" -and -not (Test-Path "results\dev\frozen_pseudo_gold_config.json")) {
    Write-Host "ERROR: results\dev\frozen_pseudo_gold_config.json was not found."
    Write-Host "Run 01_RUN_DEVELOPMENT.cmd successfully before the final test."
    exit 1
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"

function New-PythonCommand {
    param([string[]]$Arguments)
    return [pscustomobject]@{
        Executable = "python"
        Arguments  = $Arguments
    }
}

if ($Stage -eq "development") {
    $LogFile = Join-Path $PSScriptRoot "logs\development_$RunStamp.log"
    $Commands = @(
        (New-PythonCommand @("-m", "src.parse_bc5cdr", "--split", "all")),
        (New-PythonCommand @("-m", "src.build_gold_bc5cdr", "--split", "dev")),
        (New-PythonCommand @("-m", "src.scispacy_bc5cdr", "--split", "dev")),
        (New-PythonCommand @("-m", "src.biobert_bc5cdr", "--split", "dev")),
        (New-PythonCommand @("-m", "src.pubmed_bc5cdr", "--split", "dev")),
        (New-PythonCommand @("-m", "src.clinicalbert_bc5cdr", "--split", "dev")),
        (New-PythonCommand @("-m", "src.bioelectra_bc5cdr", "--split", "dev")),
        (New-PythonCommand @("-m", "src.candidate_gold", "--split", "dev")),
        (New-PythonCommand @("-m", "src.threshold_sensitivity")),
        (New-PythonCommand @("-m", "src.bc5cdr_evaluation", "--split", "dev"))
    )
    $Header = "BC5CDR DEVELOPMENT RUN"
}
else {
    $LogFile = Join-Path $PSScriptRoot "logs\final_test_$RunStamp.log"
    $Commands = @(
        (New-PythonCommand @("-m", "src.build_gold_bc5cdr", "--split", "test")),
        (New-PythonCommand @("-m", "src.scispacy_bc5cdr", "--split", "test")),
        (New-PythonCommand @("-m", "src.biobert_bc5cdr", "--split", "test")),
        (New-PythonCommand @("-m", "src.pubmed_bc5cdr", "--split", "test")),
        (New-PythonCommand @("-m", "src.clinicalbert_bc5cdr", "--split", "test")),
        (New-PythonCommand @("-m", "src.bioelectra_bc5cdr", "--split", "test")),
        (New-PythonCommand @("-m", "src.candidate_gold", "--split", "test")),
        (New-PythonCommand @("-m", "src.bc5cdr_evaluation", "--split", "test")),
        (New-PythonCommand @("-m", "src.bootstrap_significance", "--split", "test", "--resamples", "1000")),
        (New-PythonCommand @("-m", "src.cohen_kappa", "--split", "test")),
        (New-PythonCommand @("-m", "src.error_taxonomy", "--split", "test", "--models", "scispacy", "pubmedbert"))
    )
    $Header = "BC5CDR FINAL TEST RUN"
}

function Write-LoggedLine {
    param([string]$Text = "")
    Write-Host $Text
    Add-Content -LiteralPath $script:LogFile -Value $Text -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $DisplayCommand = $Executable + " " + ($Arguments -join " ")
    Write-LoggedLine ""
    Write-LoggedLine "============================================================"
    Write-LoggedLine "RUNNING: $DisplayCommand"
    Write-LoggedLine "============================================================"

    & $Executable @Arguments 2>&1 |
        Tee-Object -FilePath $script:LogFile -Append |
        ForEach-Object { Write-Host $_ }

    $ExitCode = $LASTEXITCODE
    if ($null -eq $ExitCode) {
        $ExitCode = 0
    }

    if ($ExitCode -ne 0) {
        Write-LoggedLine ""
        Write-LoggedLine "ERROR: Command failed with exit code ${ExitCode}: $DisplayCommand"
    }

    return [int]$ExitCode
}

Set-Content -LiteralPath $LogFile -Value $Header -Encoding UTF8
Write-LoggedLine "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-LoggedLine "Project: $PSScriptRoot"
Write-LoggedLine "Python: $($PythonCommand.Source)"
Write-LoggedLine "Log: $LogFile"

foreach ($Command in $Commands) {
    $Code = Invoke-LoggedCommand -Executable $Command.Executable -Arguments $Command.Arguments
    if ($Code -ne 0) {
        Write-LoggedLine ""
        Write-LoggedLine "$($Stage.ToUpper()) RUN FAILED."
        Write-LoggedLine "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host ""
        Write-Host "Check the full log: $LogFile"
        exit $Code
    }
}

Write-LoggedLine ""
if ($Stage -eq "development") {
    Write-LoggedLine "Development stage completed successfully."
    Write-LoggedLine "Frozen settings: results\dev\frozen_pseudo_gold_config.json"
}
else {
    Write-LoggedLine "Final test stage completed successfully."
    Write-LoggedLine "Results: results\test\"
}
Write-LoggedLine "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""
Write-Host "Full log: $LogFile"
exit 0
