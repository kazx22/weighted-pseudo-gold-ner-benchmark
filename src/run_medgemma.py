
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.pipeline_common import (
    ROOT,
    Stage,
    biored_dirs,
    biored_env,
    check_ollama,
    ensure_paths,
    run_stage,
)

from src.symmetric_reporting import generate as generate_symmetric_reporting

LOG_DIR = ROOT / "logs"


def _check_prerequisites() -> None:
    required = [
        ROOT / "results" / "dev" / "frozen_pseudo_gold_config.json",
        ROOT / "results" / "test" / "pseudo_gold_application.json",
        ROOT / "results" / "dev" / "cawv" / "frozen_cawv_config.json",
        ROOT / "data" / "gold" / "cawv_pseudo_gold_test_entities_bc5cdr.jsonl",
        ROOT / "results" / "biored" / "dev" / "frozen_pseudo_gold_config.json",
        ROOT / "results" / "biored" / "test" / "pseudo_gold_application.json",
        ROOT / "results" / "biored" / "dev" / "cawv" / "frozen_cawv_config.json",
        ROOT / "data" / "gold" / "biored" / "cawv_pseudo_gold_test_entities_biored.jsonl",
        ROOT / "results" / "biored_overlap_excluded" / "dev" / "frozen_pseudo_gold_config.json",
        ROOT / "results" / "biored_overlap_excluded" / "test" / "pseudo_gold_application.json",
        ROOT / "results" / "biored_overlap_excluded" / "dev" / "cawv" / "frozen_cawv_config.json",
        ROOT / "results" / "biored_overlap_excluded" / "test" / "cawv" / "cawv_evaluation.json",
    ]
    ensure_paths(
        required,
        "MedGemma needs the completed BC5CDR and BioRED runs. "
        "Run RUN_BC5CDR.cmd and RUN_BIORED.cmd first, or use RUN_ALL.cmd.",
    )


def bc5cdr_medgemma_stages() -> list[Stage]:
    proc = ROOT / "data" / "processed" / "bc5cdr"
    gold = ROOT / "data" / "gold"
    dev = ROOT / "results" / "dev"
    test = ROOT / "results" / "test"
    return [
        Stage(
            "BC5CDR MedGemma zero-shot development",
            "src.medgemma_bc5cdr",
            ("--split", "dev"),
            (proc / "medgemma_dev_entities_bc5cdr.jsonl",),
        ),
        Stage(
            "BC5CDR selective MedGemma development",
            "src.medgemma_hybrid",
            ("--split", "dev"),
            (gold / "medgemma_hybrid_dev_entities_bc5cdr.jsonl",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR MedGemma development evaluation",
            "src.medgemma_evaluation",
            ("--split", "dev"),
            (dev / "medgemma_evaluation.json",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR MedGemma zero-shot test",
            "src.medgemma_bc5cdr",
            ("--split", "test"),
            (proc / "medgemma_test_entities_bc5cdr.jsonl",),
        ),
        Stage(
            "BC5CDR selective MedGemma test",
            "src.medgemma_hybrid",
            ("--split", "test"),
            (gold / "medgemma_hybrid_test_entities_bc5cdr.jsonl",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR MedGemma test evaluation",
            "src.medgemma_evaluation",
            ("--split", "test"),
            (test / "medgemma_evaluation.json",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR MedGemma paired bootstrap",
            "src.medgemma_bootstrap",
            ("--split", "test", "--resamples", "1000", "--seed", "42"),
            (test / "medgemma_bootstrap_results.json",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR MedGemma report",
            "src.medgemma_report",
            ("--split", "both"),
            (test / "medgemma_report" / "MEDGEMMA_RESULTS_REPORT.md",),
        
            always_run=True,
        ),
    ]


def biored_medgemma_stages(variant: str) -> list[Stage]:
    proc, gold, results = biored_dirs(variant)
    env = biored_env(variant)
    label = "BioRED official" if variant == "official" else "BioRED overlap-excluded"
    return [
        Stage(
            f"{label} MedGemma zero-shot development",
            "src.biored_medgemma",
            ("--split", "dev"),
            (proc / "medgemma_dev_entities_biored.jsonl",),
            env,
        ),
        Stage(
            f"{label} selective MedGemma development",
            "src.biored_medgemma_hybrid",
            ("--split", "dev"),
            (gold / "medgemma_hybrid_dev_entities_biored.jsonl",),
            env,
        
            always_run=True,
        ),
        Stage(
            f"{label} MedGemma development evaluation",
            "src.biored_medgemma_evaluation",
            ("--split", "dev"),
            (results / "dev" / "medgemma_evaluation.json",),
            env,
        
            always_run=True,
        ),
        Stage(
            f"{label} MedGemma zero-shot test",
            "src.biored_medgemma",
            ("--split", "test"),
            (proc / "medgemma_test_entities_biored.jsonl",),
            env,
        ),
        Stage(
            f"{label} selective MedGemma test",
            "src.biored_medgemma_hybrid",
            ("--split", "test"),
            (gold / "medgemma_hybrid_test_entities_biored.jsonl",),
            env,
        
            always_run=True,
        ),
        Stage(
            f"{label} MedGemma test evaluation",
            "src.biored_medgemma_evaluation",
            ("--split", "test"),
            (results / "test" / "medgemma_evaluation.json",),
            env,
        
            always_run=True,
        ),
        Stage(
            f"{label} MedGemma paired bootstrap",
            "src.biored_medgemma_bootstrap",
            ("--resamples", "1000", "--seed", "42"),
            (results / "test" / "medgemma_bootstrap_results.json",),
            env,
        
            always_run=True,
        ),
        Stage(
            f"{label} MedGemma report",
            "src.biored_report",
            ("--split", "both"),
            (results / "test" / "biored_report" / "BIORED_MEDGEMMA_REPORT.md",),
            env,
        
            always_run=True,
        ),
    ]


def bc5cdr_cawv_medgemma_stages() -> list[Stage]:
    gold = ROOT / "data" / "gold"
    dev = ROOT / "results" / "dev" / "medgemma_cawv_hybrid"
    test = ROOT / "results" / "test" / "medgemma_cawv_hybrid"
    return [
        Stage(
            "BC5CDR CAWV + MedGemma development routing",
            "src.cawv_medgemma_hybrid",
            ("--dataset", "bc5cdr", "--split", "dev"),
            (gold / "medgemma_cawv_hybrid_dev_entities_bc5cdr.jsonl", ROOT / "results" / "dev" / "medgemma_cawv_hybrid_config.json"),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR CAWV + MedGemma development evaluation",
            "src.cawv_medgemma_analysis",
            ("--dataset", "bc5cdr", "--split", "dev"),
            (dev / "cawv_medgemma_analysis.json",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR CAWV + MedGemma test routing",
            "src.cawv_medgemma_hybrid",
            ("--dataset", "bc5cdr", "--split", "test"),
            (gold / "medgemma_cawv_hybrid_test_entities_bc5cdr.jsonl",),
        
            always_run=True,
        ),
        Stage(
            "BC5CDR CAWV vs CAWV + MedGemma paired bootstrap",
            "src.cawv_medgemma_analysis",
            ("--dataset", "bc5cdr", "--split", "test", "--resamples", "1000", "--seed", "42"),
            (test / "cawv_medgemma_analysis.json",),
        
            always_run=True,
        ),
    ]


def biored_cawv_medgemma_stages() -> list[Stage]:
    _, gold, results = biored_dirs("official")
    env = biored_env("official")
    return [
        Stage(
            "BioRED official CAWV + MedGemma development routing",
            "src.cawv_medgemma_hybrid",
            ("--dataset", "biored", "--split", "dev"),
            (gold / "medgemma_cawv_hybrid_dev_entities_biored.jsonl", results / "dev" / "medgemma_cawv_hybrid_config.json"),
            env,
        
            always_run=True,
        ),
        Stage(
            "BioRED official CAWV + MedGemma development evaluation",
            "src.cawv_medgemma_analysis",
            ("--dataset", "biored", "--split", "dev"),
            (results / "dev" / "medgemma_cawv_hybrid" / "cawv_medgemma_analysis.json",),
            env,
        
            always_run=True,
        ),
        Stage(
            "BioRED official CAWV + MedGemma test routing",
            "src.cawv_medgemma_hybrid",
            ("--dataset", "biored", "--split", "test"),
            (gold / "medgemma_cawv_hybrid_test_entities_biored.jsonl",),
            env,
        
            always_run=True,
        ),
        Stage(
            "BioRED official CAWV vs CAWV + MedGemma paired bootstrap",
            "src.cawv_medgemma_analysis",
            ("--dataset", "biored", "--split", "test", "--resamples", "1000", "--seed", "42"),
            (results / "test" / "medgemma_cawv_hybrid" / "cawv_medgemma_analysis.json",),
            env,
        
            always_run=True,
        ),
    ]


def run_pipeline(*, force: bool = False) -> Path:
    _check_prerequisites()
    check_ollama()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_medgemma_{stamp}.log"

    with log_path.open("w", encoding="utf-8") as log:
        print("MEDGEMMA STUDY", file=log)
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=log, flush=True)
        for stage in bc5cdr_medgemma_stages():
            run_stage(stage, log, force=force)
        for stage in bc5cdr_cawv_medgemma_stages():
            run_stage(stage, log, force=force)
        for variant in ("official", "overlap_excluded"):
            for stage in biored_medgemma_stages(variant):
                run_stage(stage, log, force=force)
        for stage in biored_cawv_medgemma_stages():
            run_stage(stage, log, force=force)

                                                                                   
        comparison = Stage(
            "Refresh BioRED official-vs-overlap comparison with MedGemma results",
            "src.biored_variant_compare",
            (),
            (
                ROOT
                / "results"
                / "biored_sensitivity_comparison"
                / "official_vs_overlap_excluded.json",
            ),
        )
        run_stage(comparison, log, force=True)
        run_stage(
            Stage(
                "Refresh resource and deployment summary",
                "src.resource_summary",
                (),
                (
                    ROOT / "results" / "resource_usage" / "deployment_resource_summary.csv",
                    ROOT / "results" / "resource_usage" / "cloud_cost_inputs.csv",
                ),
                always_run=True,
            ),
            log,
            force=True,
        )
        run_stage(
            Stage(
                "Refresh manuscript results transfer log",
                "src.manuscript_results_log",
                (),
                (ROOT / "results" / "MANUSCRIPT_RESULTS.log",),
                always_run=True,
            ),
            log,
            force=True,
        )

    generate_symmetric_reporting(allow_partial=True)
    print("\nMEDGEMMA COMPLETE.")
    print(f"Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only MedGemma stages.")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
