param(
    [ValidateSet("Core", "MedGemma", "All")]
    [string]$Mode = "All",
    [switch]$ForceModels
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Could not find the project virtual environment at $Python"
}

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "biored_disease_chemical_$Stamp.log"

function Write-Both([string]$Text) {
    $Text | Tee-Object -FilePath $LogFile -Append
}

function Run-Step([string]$Label, [string[]]$PythonArgs) {
    Write-Both ""
    Write-Both ("=" * 72)
    Write-Both "RUNNING: $Label"
    Write-Both ("=" * 72)

    # Windows PowerShell 5.1 can turn harmless native-program stderr output
    # (for example Python FutureWarning messages) into a terminating
    # NativeCommandError when $ErrorActionPreference is set to Stop.
    # Python warnings are therefore allowed through while the real process
    # exit code remains the authority for deciding whether a step failed.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $ExitCode = 0

    try {
        & $Python @PythonArgs 2>&1 | ForEach-Object {
            $Line = $_.ToString()
            Write-Host $Line
            Add-Content -Path $LogFile -Value $Line
        }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        throw "Step failed with exit code ${ExitCode}: $Label"
    }
}

function Run-Model-If-Needed([string]$Split, [string]$Model) {
    $Output = Join-Path $ProjectRoot "data\processed\biored\${Model}_${Split}_entities_biored.jsonl"
    if ((Test-Path $Output) -and (-not $ForceModels)) {
        Write-Both "SKIP existing prediction: $Output"
        return
    }
    Run-Step "BioRED $Split $Model" @("-m", "src.biored_models", "--split", $Split, "--model", $Model)
}

function Run-Core {
    Run-Step "Parse BioRED dev + test" @("-m", "src.parse_biored", "--split", "all")

    foreach ($Model in @("scispacy", "biobert", "pubmedbert", "clinicalbert", "bioelectra")) {
        Run-Model-If-Needed "dev" $Model
    }
    Run-Step "Fit BioRED development weights + threshold" @("-m", "src.biored_candidate_gold", "--split", "dev")
    Run-Step "Plot BioRED development threshold sweep" @("-m", "src.biored_threshold_sensitivity")
    Run-Step "Evaluate BioRED development" @("-m", "src.biored_evaluation", "--split", "dev")

    foreach ($Model in @("scispacy", "biobert", "pubmedbert", "clinicalbert", "bioelectra")) {
        Run-Model-If-Needed "test" $Model
    }
    Run-Step "Apply frozen BioRED settings to test" @("-m", "src.biored_candidate_gold", "--split", "test")
    Run-Step "Evaluate BioRED test" @("-m", "src.biored_evaluation", "--split", "test")
    Run-Step "BioRED test bootstrap" @("-m", "src.biored_bootstrap", "--resamples", "1000")
    Run-Step "BioRED test Cohen kappa" @("-m", "src.biored_kappa", "--split", "test")
}

function Run-MedGemma {
    $WeightedDev = Join-Path $ProjectRoot "data\gold\biored\weighted_pseudo_gold_dev_entities_biored.jsonl"
    $WeightedTest = Join-Path $ProjectRoot "data\gold\biored\weighted_pseudo_gold_test_entities_biored.jsonl"
    if (-not (Test-Path $WeightedDev) -or -not (Test-Path $WeightedTest)) {
        throw "BioRED core outputs are missing. Run 07_RUN_BIORED_CORE.cmd first, or use 09_RUN_BIORED_ALL.cmd."
    }

    Run-Step "BioRED MedGemma smoke test (1 dev document)" @("-m", "src.biored_medgemma", "--split", "dev", "--limit", "1")
    Run-Step "BioRED MedGemma zero-shot development" @("-m", "src.biored_medgemma", "--split", "dev")
    Run-Step "BioRED selective MedGemma development" @("-m", "src.biored_medgemma_hybrid", "--split", "dev")
    Run-Step "Evaluate BioRED MedGemma development" @("-m", "src.biored_medgemma_evaluation", "--split", "dev")

    Run-Step "BioRED MedGemma zero-shot test" @("-m", "src.biored_medgemma", "--split", "test")
    Run-Step "BioRED selective MedGemma test" @("-m", "src.biored_medgemma_hybrid", "--split", "test")
    Run-Step "Evaluate BioRED MedGemma test" @("-m", "src.biored_medgemma_evaluation", "--split", "test")
    Run-Step "Bootstrap BioRED MedGemma test" @("-m", "src.biored_medgemma_bootstrap", "--resamples", "1000")
    Run-Step "Generate BioRED paper figures + report" @("-m", "src.biored_report", "--split", "both")
}

Write-Both "BIORED DISEASE/CHEMICAL EXTERNAL VALIDATION"
Write-Both "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Both "Mode: $Mode"
Write-Both "Project: $ProjectRoot"

if ($Mode -eq "Core" -or $Mode -eq "All") {
    Run-Core
}
if ($Mode -eq "MedGemma" -or $Mode -eq "All") {
    Run-MedGemma
}

Write-Both ""
Write-Both "BIORED PIPELINE COMPLETED SUCCESSFULLY."
Write-Both "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Both "Main test results: results\biored\test\evaluation_results.json"
Write-Both "MedGemma results: results\biored\test\medgemma_evaluation.json"
Write-Both "Report: results\biored\test\biored_report\BIORED_MEDGEMMA_REPORT.md"
Write-Both "Log: $LogFile"