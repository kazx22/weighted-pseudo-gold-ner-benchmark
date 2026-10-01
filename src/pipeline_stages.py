"""Shared stage definitions for the three top-level experiment runners.

This file defines orchestration only. Every scientific task remains executable
as its own `python -m src.<module>` command for easy debugging and reruns.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_KEYS = ("scispacy", "biobert", "pubmedbert", "clinicalbert", "d4data")
MEDGEMMA_MODEL = "medgemma1.5:4b-it-q4_K_M"


@dataclass
class Stage:
    label: str
    args: list[str]
    outputs: list[Path]
    env: dict[str, str] | None = None


def variant_dirs(variant: str) -> tuple[Path, Path, Path]:
    if variant == "official":
        return (
            ROOT / "data" / "processed" / "biored",
            ROOT / "data" / "gold" / "biored",
            ROOT / "results" / "biored",
        )
    if variant == "overlap_excluded":
        return (
            ROOT / "data" / "processed" / "biored_overlap_excluded",
            ROOT / "data" / "gold" / "biored_overlap_excluded",
            ROOT / "results" / "biored_overlap_excluded",
        )
    raise ValueError(f"Unknown BioRED variant: {variant}")


def biored_env(variant: str) -> dict[str, str]:
    return {"BIORED_VARIANT": variant}


def stage_done(stage: Stage) -> bool:
    return bool(stage.outputs) and all(path.exists() for path in stage.outputs)


def run_stage(stage: Stage, log_handle, force: bool = False) -> None:
    if not force and stage_done(stage):
        message = f"[SKIP] {stage.label}"
        print(message)
        print(message, file=log_handle, flush=True)
        return

    command = [sys.executable, *stage.args]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if stage.env:
        env.update(stage.env)

    divider = "=" * 78
    header = f"{divider}\nRUNNING: {stage.label}\nCMD: {' '.join(command)}\n{divider}"
    print("\n" + header)
    print("\n" + header, file=log_handle, flush=True)

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        print(line, end="", file=log_handle, flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Stage failed ({code}): {stage.label}")

    missing = [str(path) for path in stage.outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Stage completed but expected output(s) are missing: {stage.label}\n  "
            + "\n  ".join(missing)
        )


def check_raw_inputs() -> None:
    required = [
        ROOT / "data" / "raw" / "BIORED.zip",
        ROOT / "data" / "raw" / "bc5cdr" / "CDR_TrainingSet.PubTator.txt",
        ROOT / "data" / "raw" / "bc5cdr" / "CDR_DevelopmentSet.PubTator.txt",
        ROOT / "data" / "raw" / "bc5cdr" / "CDR_TestSet.PubTator.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raw input(s):\n  " + "\n  ".join(missing))


def check_ollama() -> None:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. Start Ollama and rerun."
        ) from exc
    installed = {
        str(item.get("name") or item.get("model") or "") for item in payload.get("models", [])
    }
    if MEDGEMMA_MODEL not in installed:
        raise RuntimeError(
            f"Required Ollama model is not installed: {MEDGEMMA_MODEL}\n"
            f"Install once with: ollama pull {MEDGEMMA_MODEL}"
        )


def bc5_core_stages() -> list[Stage]:
    proc = ROOT / "data" / "processed" / "bc5cdr"
    gold = ROOT / "data" / "gold"
    dev = ROOT / "results" / "dev"
    test = ROOT / "results" / "test"
    module_for = {
        "scispacy": "src.scispacy_bc5cdr",
        "biobert": "src.biobert_bc5cdr",
        "pubmedbert": "src.pubmed_bc5cdr",
        "clinicalbert": "src.clinicalbert_bc5cdr",
        "d4data": "src.d4data_bc5cdr",
    }

    stages = [
        Stage("BC5CDR parse raw corpus", ["-m", "src.parse_bc5cdr", "--split", "all"], [proc / "bc5cdr_dev_docs.jsonl", proc / "bc5cdr_test_docs.jsonl"]),
        Stage("BC5CDR development BIO gold", ["-m", "src.build_gold_bc5cdr", "--split", "dev"], [gold / "bc5cdr_dev_gold_bio.jsonl"]),
    ]
    for model in MODEL_KEYS:
        stages.append(Stage(f"BC5CDR dev model: {model}", ["-m", module_for[model], "--split", "dev"], [proc / f"{model}_dev_entities_bc5cdr.jsonl"]))
    stages += [
        Stage("BC5CDR fit development weights + threshold", ["-m", "src.candidate_gold", "--split", "dev"], [dev / "frozen_pseudo_gold_config.json"]),
        Stage("BC5CDR development threshold sensitivity", ["-m", "src.threshold_sensitivity"], [dev / "figures" / "threshold_sensitivity_curve.png"]),
        Stage("BC5CDR development evaluation", ["-m", "src.bc5cdr_evaluation", "--split", "dev"], [dev / "evaluation_results.json"]),
        Stage("BC5CDR test BIO gold", ["-m", "src.build_gold_bc5cdr", "--split", "test"], [gold / "bc5cdr_test_gold_bio.jsonl"]),
    ]
    for model in MODEL_KEYS:
        stages.append(Stage(f"BC5CDR test model: {model}", ["-m", module_for[model], "--split", "test"], [proc / f"{model}_test_entities_bc5cdr.jsonl"]))
    stages += [
        Stage("BC5CDR apply frozen weighted configuration to test", ["-m", "src.candidate_gold", "--split", "test"], [test / "pseudo_gold_application.json"]),
        Stage("BC5CDR held-out test evaluation", ["-m", "src.bc5cdr_evaluation", "--split", "test"], [test / "evaluation_results.json"]),
        Stage("BC5CDR paired bootstrap + Holm correction", ["-m", "src.bootstrap_significance", "--split", "test", "--resamples", "1000"], [test / "bootstrap_results.json"]),
        Stage("BC5CDR token-level Cohen kappa", ["-m", "src.cohen_kappa", "--split", "test"], [test / "cohen_kappa_results.json"]),
        Stage("BC5CDR qualitative error taxonomy", ["-m", "src.error_taxonomy", "--split", "test", "--models", "scispacy", "pubmedbert"], [test / "error_taxonomy" / "scispacy_error_counts.txt", test / "error_taxonomy" / "pubmedbert_error_counts.txt"]),
        Stage("BC5CDR tuned-unweighted + leave-one-model-out controls", ["-m", "src.method_controls", "--dataset", "bc5cdr", "--resamples", "1000"], [test / "method_controls" / "method_controls.json"]),
        Stage("BC5CDR development-gold budget sensitivity", ["-m", "src.development_gold_sensitivity", "--dataset", "bc5cdr", "--seeds", "10"], [test / "development_gold_sensitivity" / "development_gold_sensitivity.json"]),
    ]
    return stages


def biored_core_stages(variant: str) -> list[Stage]:
    proc, _gold, results = variant_dirs(variant)
    env = biored_env(variant)
    stages: list[Stage] = []

    if variant == "official":
        stages.append(Stage("BioRED official parse dev + test", ["-m", "src.parse_biored", "--split", "all"], [proc / "biored_dev_docs.jsonl", proc / "biored_test_docs.jsonl"], env))
    else:
        stages.append(Stage("BioRED remove all BC5CDR-overlapping PMIDs", ["-m", "src.biored_overlap_exclusion"], [proc / "biored_dev_docs.jsonl", proc / "biored_test_docs.jsonl", results / "overlap_audit.json"]))

    for split in ("dev", "test"):
        for model in MODEL_KEYS:
            stages.append(Stage(f"BioRED {variant} {split} model: {model}", ["-m", "src.biored_models", "--split", split, "--model", model], [proc / f"{model}_{split}_entities_biored.jsonl"], env))

        if split == "dev":
            stages += [
                Stage(f"BioRED {variant} fit dev weights + threshold", ["-m", "src.biored_candidate_gold", "--split", "dev"], [results / "dev" / "frozen_pseudo_gold_config.json"], env),
                Stage(f"BioRED {variant} development threshold sensitivity", ["-m", "src.biored_threshold_sensitivity"], [results / "dev" / "figures" / "biored_threshold_sensitivity_curve.png"], env),
                Stage(f"BioRED {variant} development evaluation", ["-m", "src.biored_evaluation", "--split", "dev"], [results / "dev" / "evaluation_results.json"], env),
            ]
        else:
            stages += [
                Stage(f"BioRED {variant} apply frozen dev settings to test", ["-m", "src.biored_candidate_gold", "--split", "test"], [results / "test" / "pseudo_gold_application.json"], env),
                Stage(f"BioRED {variant} held-out test evaluation", ["-m", "src.biored_evaluation", "--split", "test"], [results / "test" / "evaluation_results.json"], env),
                Stage(f"BioRED {variant} paired bootstrap + Holm correction", ["-m", "src.biored_bootstrap", "--resamples", "1000"], [results / "test" / "bootstrap_results.json"], env),
                Stage(f"BioRED {variant} token-level Cohen kappa", ["-m", "src.biored_kappa", "--split", "test"], [results / "test" / "cohen_kappa_results.json"], env),
                Stage(f"BioRED {variant} qualitative error taxonomy", ["-m", "src.biored_error_taxonomy", "--split", "test", "--models", "scispacy", "pubmedbert"], [results / "test" / "error_taxonomy" / "biored_error_taxonomy_summary.txt"], env),
                Stage(f"BioRED {variant} tuned-unweighted + leave-one-model-out controls", ["-m", "src.method_controls", "--dataset", "biored", "--resamples", "1000"], [results / "test" / "method_controls" / "method_controls.json"], env),
                Stage(f"BioRED {variant} development-gold budget sensitivity", ["-m", "src.development_gold_sensitivity", "--dataset", "biored", "--seeds", "10"], [results / "test" / "development_gold_sensitivity" / "development_gold_sensitivity.json"], env),
            ]
    return stages


def bc5_medgemma_stages() -> list[Stage]:
    proc = ROOT / "data" / "processed" / "bc5cdr"
    gold = ROOT / "data" / "gold"
    dev = ROOT / "results" / "dev"
    test = ROOT / "results" / "test"
    return [
        Stage("BC5CDR MedGemma zero-shot dev", ["-m", "src.medgemma_bc5cdr", "--split", "dev"], [proc / "medgemma_dev_entities_bc5cdr.jsonl"]),
        Stage("BC5CDR selective MedGemma dev", ["-m", "src.medgemma_hybrid", "--split", "dev"], [gold / "medgemma_hybrid_dev_entities_bc5cdr.jsonl"]),
        Stage("BC5CDR MedGemma evaluation dev", ["-m", "src.medgemma_evaluation", "--split", "dev"], [dev / "medgemma_evaluation.json"]),
        Stage("BC5CDR MedGemma zero-shot test", ["-m", "src.medgemma_bc5cdr", "--split", "test"], [proc / "medgemma_test_entities_bc5cdr.jsonl"]),
        Stage("BC5CDR selective MedGemma test", ["-m", "src.medgemma_hybrid", "--split", "test"], [gold / "medgemma_hybrid_test_entities_bc5cdr.jsonl"]),
        Stage("BC5CDR MedGemma evaluation test", ["-m", "src.medgemma_evaluation", "--split", "test"], [test / "medgemma_evaluation.json"]),
        Stage("BC5CDR MedGemma paired bootstrap", ["-m", "src.medgemma_bootstrap", "--split", "test", "--resamples", "1000"], [test / "medgemma_bootstrap_results.json"]),
        Stage("BC5CDR MedGemma report", ["-m", "src.medgemma_report", "--split", "both"], [test / "medgemma_report" / "MEDGEMMA_RESULTS_REPORT.md"]),
    ]


def biored_medgemma_stages(variant: str) -> list[Stage]:
    proc, gold, results = variant_dirs(variant)
    env = biored_env(variant)
    return [
        Stage(f"BioRED {variant} MedGemma zero-shot dev", ["-m", "src.biored_medgemma", "--split", "dev"], [proc / "medgemma_dev_entities_biored.jsonl"], env),
        Stage(f"BioRED {variant} selective MedGemma dev", ["-m", "src.biored_medgemma_hybrid", "--split", "dev"], [gold / "medgemma_hybrid_dev_entities_biored.jsonl"], env),
        Stage(f"BioRED {variant} MedGemma evaluation dev", ["-m", "src.biored_medgemma_evaluation", "--split", "dev"], [results / "dev" / "medgemma_evaluation.json"], env),
        Stage(f"BioRED {variant} MedGemma zero-shot test", ["-m", "src.biored_medgemma", "--split", "test"], [proc / "medgemma_test_entities_biored.jsonl"], env),
        Stage(f"BioRED {variant} selective MedGemma test", ["-m", "src.biored_medgemma_hybrid", "--split", "test"], [gold / "medgemma_hybrid_test_entities_biored.jsonl"], env),
        Stage(f"BioRED {variant} MedGemma evaluation test", ["-m", "src.biored_medgemma_evaluation", "--split", "test"], [results / "test" / "medgemma_evaluation.json"], env),
        Stage(f"BioRED {variant} MedGemma paired bootstrap", ["-m", "src.biored_medgemma_bootstrap", "--resamples", "1000"], [results / "test" / "medgemma_bootstrap_results.json"], env),
        Stage(f"BioRED {variant} MedGemma report", ["-m", "src.biored_report", "--split", "both"], [results / "test" / "biored_report" / "BIORED_MEDGEMMA_REPORT.md"], env),
    ]


def conventional_prerequisites() -> list[Path]:
    _, _, official = variant_dirs("official")
    _, _, clean = variant_dirs("overlap_excluded")
    return [
        ROOT / "results" / "dev" / "frozen_pseudo_gold_config.json",
        ROOT / "results" / "test" / "evaluation_results.json",
        official / "dev" / "frozen_pseudo_gold_config.json",
        official / "test" / "evaluation_results.json",
        clean / "dev" / "frozen_pseudo_gold_config.json",
        clean / "test" / "evaluation_results.json",
    ]


def require_conventional_results() -> None:
    missing = [str(path) for path in conventional_prerequisites() if not path.exists()]
    if missing:
        raise RuntimeError(
            "MedGemma requires completed conventional/calibration outputs.\n"
            "Run RUN_TRANSFORMERS.cmd first. Missing:\n  " + "\n  ".join(missing)
        )



def conventional_comparison_stage() -> Stage:
    return Stage(
        "Compare official vs overlap-excluded BioRED conventional results",
        ["-m", "src.biored_conventional_compare"],
        [ROOT / "results" / "biored_sensitivity_comparison" / "conventional_official_vs_overlap_excluded.json"],
    )

def comparison_stage() -> Stage:
    return Stage(
        "Compare official BioRED with overlap-excluded sensitivity analysis",
        ["-m", "src.biored_variant_compare"],
        [ROOT / "results" / "biored_sensitivity_comparison" / "official_vs_overlap_excluded.json"],
    )
