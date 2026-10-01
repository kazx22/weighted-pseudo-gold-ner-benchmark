"""Run every non-MedGemma experiment in the study.

Despite the historical command name RUN_TRANSFORMERS.cmd, this runner includes
all five conventional systems, including scispaCy. Scientific stages remain in
separate modules and can be rerun individually if troubleshooting is needed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.pipeline_common import (
    MODEL_KEYS,
    ROOT,
    Stage,
    biored_dirs,
    biored_env,
    check_raw_inputs,
    run_stage,
)

LOG_DIR = ROOT / "logs"


def bc5cdr_stages() -> list[Stage]:
    proc = ROOT / "data" / "processed" / "bc5cdr"
    gold = ROOT / "data" / "gold"
    dev = ROOT / "results" / "dev"
    test = ROOT / "results" / "test"
    modules = {
        "scispacy": "src.scispacy_bc5cdr",
        "biobert": "src.biobert_bc5cdr",
        "pubmedbert": "src.pubmed_bc5cdr",
        "clinicalbert": "src.clinicalbert_bc5cdr",
        "d4data": "src.d4data_bc5cdr",
    }

    stages = [
        Stage(
            "BC5CDR parse raw corpus",
            "src.parse_bc5cdr",
            ("--split", "all"),
            (
                proc / "bc5cdr_dev_docs.jsonl",
                proc / "bc5cdr_dev_entities.jsonl",
                proc / "bc5cdr_test_docs.jsonl",
                proc / "bc5cdr_test_entities.jsonl",
            ),
        ),
        Stage(
            "BC5CDR development BIO gold",
            "src.build_gold_bc5cdr",
            ("--split", "dev"),
            (gold / "bc5cdr_dev_gold_bio.jsonl",),
        ),
    ]

    for model in MODEL_KEYS:
        stages.append(
            Stage(
                f"BC5CDR development model: {model}",
                modules[model],
                ("--split", "dev"),
                (proc / f"{model}_dev_entities_bc5cdr.jsonl",),
            )
        )

    stages.extend(
        [
            Stage(
                "BC5CDR fit development weights and weighted threshold",
                "src.candidate_gold",
                ("--split", "dev"),
                (dev / "frozen_pseudo_gold_config.json",),
            ),
            Stage(
                "BC5CDR development threshold sensitivity",
                "src.threshold_sensitivity",
                (),
                (dev / "figures" / "threshold_sensitivity_curve.png",),
            ),
            Stage(
                "BC5CDR development evaluation",
                "src.bc5cdr_evaluation",
                ("--split", "dev"),
                (dev / "evaluation_results.json",),
            ),
            Stage(
                "BC5CDR test BIO gold",
                "src.build_gold_bc5cdr",
                ("--split", "test"),
                (gold / "bc5cdr_test_gold_bio.jsonl",),
            ),
        ]
    )

    for model in MODEL_KEYS:
        stages.append(
            Stage(
                f"BC5CDR test model: {model}",
                modules[model],
                ("--split", "test"),
                (proc / f"{model}_test_entities_bc5cdr.jsonl",),
            )
        )

    stages.extend(
        [
            Stage(
                "BC5CDR apply frozen development configuration to test",
                "src.candidate_gold",
                ("--split", "test"),
                (test / "pseudo_gold_application.json",),
            ),
            Stage(
                "BC5CDR held-out test evaluation",
                "src.bc5cdr_evaluation",
                ("--split", "test"),
                (test / "evaluation_results.json",),
            ),
            Stage(
                "BC5CDR paired bootstrap and Holm correction",
                "src.bootstrap_significance",
                ("--split", "test", "--resamples", "1000"),
                (test / "bootstrap_results.json",),
            ),
            Stage(
                "BC5CDR token-level Cohen kappa",
                "src.cohen_kappa",
                ("--split", "test"),
                (test / "cohen_kappa_results.json",),
            ),
            Stage(
                "BC5CDR qualitative error taxonomy",
                "src.error_taxonomy",
                ("--split", "test", "--models", "scispacy", "pubmedbert"),
                (
                    test / "error_taxonomy" / "scispacy_error_counts.txt",
                    test / "error_taxonomy" / "pubmedbert_error_counts.txt",
                ),
            ),
            Stage(
                "BC5CDR tuned-unweighted control and leave-one-model-out ablation",
                "src.method_controls",
                ("--dataset", "bc5cdr", "--resamples", "1000"),
                (test / "method_controls" / "method_controls.json",),
            ),
            Stage(
                "BC5CDR 25/50/75/100 percent development-gold sensitivity",
                "src.development_gold_sensitivity",
                ("--dataset", "bc5cdr", "--seeds", "10"),
                (
                    test
                    / "development_gold_sensitivity"
                    / "development_gold_sensitivity.json",
                ),
            ),
        ]
    )
    return stages


def biored_official_stages() -> list[Stage]:
    proc, _, results = biored_dirs("official")
    env = biored_env("official")
    stages = [
        Stage(
            "BioRED official parse development and test",
            "src.parse_biored",
            ("--split", "all"),
            (
                proc / "biored_dev_docs.jsonl",
                proc / "biored_dev_entities.jsonl",
                proc / "biored_test_docs.jsonl",
                proc / "biored_test_entities.jsonl",
            ),
            env,
        )
    ]

    for split in ("dev", "test"):
        for model in MODEL_KEYS:
            stages.append(
                Stage(
                    f"BioRED official {split} model: {model}",
                    "src.biored_models",
                    ("--split", split, "--model", model),
                    (proc / f"{model}_{split}_entities_biored.jsonl",),
                    env,
                )
            )

        if split == "dev":
            stages.extend(
                [
                    Stage(
                        "BioRED official fit development weights and threshold",
                        "src.biored_candidate_gold",
                        ("--split", "dev"),
                        (results / "dev" / "frozen_pseudo_gold_config.json",),
                        env,
                    ),
                    Stage(
                        "BioRED official development threshold sensitivity",
                        "src.biored_threshold_sensitivity",
                        (),
                        (
                            results
                            / "dev"
                            / "figures"
                            / "biored_threshold_sensitivity_curve.png",
                        ),
                        env,
                    ),
                    Stage(
                        "BioRED official development evaluation",
                        "src.biored_evaluation",
                        ("--split", "dev"),
                        (results / "dev" / "evaluation_results.json",),
                        env,
                    ),
                ]
            )
        else:
            stages.extend(_biored_test_analysis_stages("official"))
    return stages


def _biored_test_analysis_stages(variant: str) -> list[Stage]:
    _, _, results = biored_dirs(variant)
    env = biored_env(variant)
    label = "BioRED official" if variant == "official" else "BioRED overlap-excluded"
    return [
        Stage(
            f"{label} apply frozen development configuration to test",
            "src.biored_candidate_gold",
            ("--split", "test"),
            (results / "test" / "pseudo_gold_application.json",),
            env,
        ),
        Stage(
            f"{label} held-out test evaluation",
            "src.biored_evaluation",
            ("--split", "test"),
            (results / "test" / "evaluation_results.json",),
            env,
        ),
        Stage(
            f"{label} paired bootstrap and Holm correction",
            "src.biored_bootstrap",
            ("--resamples", "1000"),
            (results / "test" / "bootstrap_results.json",),
            env,
        ),
        Stage(
            f"{label} token-level Cohen kappa",
            "src.biored_kappa",
            ("--split", "test"),
            (results / "test" / "cohen_kappa_results.json",),
            env,
        ),
        Stage(
            f"{label} qualitative error taxonomy",
            "src.biored_error_taxonomy",
            ("--split", "test", "--models", "scispacy", "pubmedbert"),
            (results / "test" / "error_taxonomy" / "biored_error_taxonomy_summary.txt",),
            env,
        ),
        Stage(
            f"{label} tuned-unweighted control and leave-one-model-out ablation",
            "src.method_controls",
            ("--dataset", "biored", "--resamples", "1000"),
            (results / "test" / "method_controls" / "method_controls.json",),
            env,
        ),
        Stage(
            f"{label} 25/50/75/100 percent development-gold sensitivity",
            "src.development_gold_sensitivity",
            ("--dataset", "biored", "--seeds", "10"),
            (
                results
                / "test"
                / "development_gold_sensitivity"
                / "development_gold_sensitivity.json",
            ),
            env,
        ),
    ]


def biored_overlap_excluded_stages() -> list[Stage]:
    proc, _, results = biored_dirs("overlap_excluded")
    env = biored_env("overlap_excluded")
    stages = [
        Stage(
            "BioRED remove every PMID shared with any BC5CDR split",
            "src.biored_overlap_exclusion",
            (),
            (
                proc / "biored_dev_docs.jsonl",
                proc / "biored_dev_entities.jsonl",
                proc / "biored_test_docs.jsonl",
                proc / "biored_test_entities.jsonl",
                results / "overlap_audit.json",
            ),
        ),
        Stage(
            "BioRED overlap-excluded reuse official development model predictions",
            "src.biored_filter_predictions",
            ("--split", "dev"),
            tuple(proc / f"{model}_dev_entities_biored.jsonl" for model in MODEL_KEYS),
        ),
        Stage(
            "BioRED overlap-excluded fit NEW development weights and threshold",
            "src.biored_candidate_gold",
            ("--split", "dev"),
            (results / "dev" / "frozen_pseudo_gold_config.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded development threshold sensitivity",
            "src.biored_threshold_sensitivity",
            (),
            (
                results
                / "dev"
                / "figures"
                / "biored_threshold_sensitivity_curve.png",
            ),
            env,
        ),
        Stage(
            "BioRED overlap-excluded development evaluation",
            "src.biored_evaluation",
            ("--split", "dev"),
            (results / "dev" / "evaluation_results.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded reuse official test model predictions",
            "src.biored_filter_predictions",
            ("--split", "test"),
            tuple(proc / f"{model}_test_entities_biored.jsonl" for model in MODEL_KEYS),
        ),
    ]
    stages.extend(_biored_test_analysis_stages("overlap_excluded"))
    return stages


def comparison_stage() -> Stage:
    # RUN_TRANSFORMERS must not depend on MedGemma outputs.
    # This conventional-only comparison is refreshed later by RUN_MEDGEMMA
    # using src.biored_variant_compare once MedGemma outputs exist.
    return Stage(
        "Compare official vs overlap-excluded BioRED conventional results",
        "src.biored_conventional_compare",
        (),
        (
            ROOT
            / "results"
            / "biored_sensitivity_comparison"
            / "conventional_official_vs_overlap_excluded.json",
        ),
    )


def run_pipeline(*, force: bool = False) -> Path:
    check_raw_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_transformers_{stamp}.log"

    with log_path.open("w", encoding="utf-8") as log:
        print("CONVENTIONAL NER + METHOD CONTROLS", file=log)
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=log, flush=True)
        for stage in bc5cdr_stages():
            run_stage(stage, log, force=force)
        for stage in biored_official_stages():
            run_stage(stage, log, force=force)
        for stage in biored_overlap_excluded_stages():
            run_stage(stage, log, force=force)
        run_stage(comparison_stage(), log, force=True)

    print("\nCONVENTIONAL/METHOD PIPELINE COMPLETED.")
    print(f"Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BC5CDR + both BioRED conventional/method experiments."
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
