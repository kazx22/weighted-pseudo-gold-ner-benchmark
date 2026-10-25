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

from src.symmetric_reporting import generate as generate_symmetric_reporting

LOG_DIR = ROOT / "logs"


def official_stages() -> list[Stage]:
    proc, gold, results = biored_dirs("official")
    env = biored_env("official")
    items: list[Stage] = [
        Stage(
            "BioRED parse development and test",
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
            items.append(
                Stage(
                    f"BioRED {split} model: {model}",
                    "src.biored_models",
                    ("--split", split, "--model", model),
                    (proc / f"{model}_{split}_entities_biored.jsonl",),
                    env,
                    resource_required=(split == "test"),
                )
            )

        items.append(
            Stage(
                f"BioRED validate {split} predictions",
                "src.span_integrity_audit",
                ("--dataset", "biored", "--split", split, "--top", "50", "--fail-on-integrity-error"),
                (results / split / "span_audit" / "span_integrity_audit.json",),
                env,
            )
        )

        if split == "dev":
            items.extend(
                [
                    Stage(
                        "BioRED fit weighted voting on development",
                        "src.biored_candidate_gold",
                        ("--split", "dev"),
                        (results / "dev" / "frozen_pseudo_gold_config.json",),
                        env,
                    ),
                    Stage(
                        "BioRED weighted-threshold sensitivity",
                        "src.biored_threshold_sensitivity",
                        (),
                        (results / "dev" / "figures" / "biored_threshold_sensitivity_curve.png",),
                        env,
                    ),
                    Stage(
                        "BioRED development evaluation",
                        "src.biored_evaluation",
                        ("--split", "dev"),
                        (results / "dev" / "evaluation_results.json",),
                        env,
                    ),
                ]
            )
        else:
            items.extend(
                [
                    Stage(
                        "BioRED apply frozen weighted voting to test",
                        "src.biored_candidate_gold",
                        ("--split", "test"),
                        (results / "test" / "pseudo_gold_application.json",),
                        env,
                    ),
                    Stage(
                        "BioRED test evaluation",
                        "src.biored_evaluation",
                        ("--split", "test"),
                        (results / "test" / "evaluation_results.json",),
                        env,
                    ),
                    Stage(
                        "BioRED weighted-voting bootstrap",
                        "src.biored_bootstrap",
                        ("--resamples", "1000", "--seed", "42"),
                        (results / "test" / "bootstrap_results.json",),
                        env,
                    ),
                    Stage(
                        "BioRED Cohen kappa",
                        "src.biored_kappa",
                        ("--split", "test"),
                        (
                            results / "test" / "cohen_kappa_results.json",
                            results / "test" / "kappa_alignment_diagnostics.json",
                        ),
                        env,
                    ),
                    Stage(
                        "BioRED kappa diagnostics",
                        "src.biored_kappa_diagnostic",
                        ("--split", "test"),
                        (results / "test" / "kappa_diagnostic" / "kappa_diagnostic.json",),
                        env,
                    ),
                    Stage(
                        "BioRED error taxonomy",
                        "src.biored_error_taxonomy",
                        ("--split", "test", "--models", *MODEL_KEYS),
                        (
                            results / "test" / "error_taxonomy" / "biored_error_taxonomy_summary.txt",
                            results / "test" / "error_taxonomy" / "scispacy_error_counts.txt",
                            results / "test" / "error_taxonomy" / "biobert_error_counts.txt",
                            results / "test" / "error_taxonomy" / "pubmedbert_error_counts.txt",
                            results / "test" / "error_taxonomy" / "clinicalbert_error_counts.txt",
                            results / "test" / "error_taxonomy" / "d4data_error_counts.txt",
                            results / "test" / "error_taxonomy" / "taxonomy_summary.json",
                        ),
                        env,
                    ),
                    Stage(
                        "BioRED relaxed-overlap evaluation",
                        "src.relaxed_overlap_evaluation",
                        ("--dataset", "biored", "--split", "test"),
                        (results / "test" / "relaxed_overlap" / "relaxed_overlap_metrics.csv",),
                        env,
                    ),
                    Stage(
                        "BioRED tuned-unweighted and leave-one-model-out controls",
                        "src.method_controls",
                        ("--dataset", "biored", "--resamples", "1000", "--seed", "42"),
                        (results / "test" / "method_controls" / "method_controls.json",),
                        env,
                    ),
                    Stage(
                        "BioRED development-gold sensitivity",
                        "src.development_gold_sensitivity",
                        ("--dataset", "biored", "--seeds", "10"),
                        (results / "test" / "development_gold_sensitivity" / "development_gold_sensitivity.json",),
                        env,
                    ),
                ]
            )

    items.extend(
        [
            Stage(
                "BioRED CAWV fit on development",
                "src.biored_content_aware_voting",
                ("--split", "dev"),
                (
                    results / "dev" / "cawv" / "frozen_cawv_config.json",
                    results / "dev" / "figures" / "cawv_alpha_sensitivity.png",
                    results / "dev" / "figures" / "cawv_threshold_curve_selected_alpha.png",
                ),
                env,
            ),
            Stage(
                "BioRED CAWV apply frozen settings to test",
                "src.biored_content_aware_voting",
                ("--split", "test"),
                (gold / "cawv_pseudo_gold_test_entities_biored.jsonl",),
                env,
            ),
            Stage(
                "BioRED CAWV evaluation and bootstrap",
                "src.biored_cawv_evaluation",
                ("--split", "test", "--resamples", "1000", "--seed", "42"),
                (results / "test" / "cawv" / "cawv_evaluation.json",),
                env,
            ),
            Stage(
                "BioRED CAWV ablation and class-wise baseline",
                "src.cawv_ablation",
                ("--dataset", "biored", "--resamples", "1000", "--seed", "42"),
                (results / "test" / "cawv" / "ablation" / "cawv_ablation_results.json",),
                env,
            ),
        ]
    )
    return items


def overlap_sensitivity_stages() -> list[Stage]:
    proc, gold, results = biored_dirs("overlap_excluded")
    env = biored_env("overlap_excluded")
    items: list[Stage] = [
        Stage(
            "BioRED overlap-excluded corpus",
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
            "BioRED overlap-excluded development predictions",
            "src.biored_filter_predictions",
            ("--split", "dev"),
            tuple(proc / f"{model}_dev_entities_biored.jsonl" for model in MODEL_KEYS),
        ),
        Stage(
            "BioRED overlap-excluded validate development predictions",
            "src.span_integrity_audit",
            ("--dataset", "biored", "--split", "dev", "--top", "50", "--fail-on-integrity-error"),
            (results / "dev" / "span_audit" / "span_integrity_audit.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded fit development weights",
            "src.biored_candidate_gold",
            ("--split", "dev"),
            (results / "dev" / "frozen_pseudo_gold_config.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded threshold sensitivity",
            "src.biored_threshold_sensitivity",
            (),
            (results / "dev" / "figures" / "biored_threshold_sensitivity_curve.png",),
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
            "BioRED overlap-excluded CAWV fit on development",
            "src.biored_content_aware_voting",
            ("--split", "dev"),
            (
                results / "dev" / "cawv" / "frozen_cawv_config.json",
                results / "dev" / "figures" / "cawv_alpha_sensitivity.png",
                results / "dev" / "figures" / "cawv_threshold_curve_selected_alpha.png",
            ),
            env,
        ),
        Stage(
            "BioRED overlap-excluded test predictions",
            "src.biored_filter_predictions",
            ("--split", "test"),
            tuple(proc / f"{model}_test_entities_biored.jsonl" for model in MODEL_KEYS),
        ),
        Stage(
            "BioRED overlap-excluded validate test predictions",
            "src.span_integrity_audit",
            ("--dataset", "biored", "--split", "test", "--top", "50", "--fail-on-integrity-error"),
            (results / "test" / "span_audit" / "span_integrity_audit.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded apply frozen settings to test",
            "src.biored_candidate_gold",
            ("--split", "test"),
            (results / "test" / "pseudo_gold_application.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded test evaluation",
            "src.biored_evaluation",
            ("--split", "test"),
            (results / "test" / "evaluation_results.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded CAWV apply frozen settings to test",
            "src.biored_content_aware_voting",
            ("--split", "test"),
            (gold / "cawv_pseudo_gold_test_entities_biored.jsonl",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded CAWV evaluation and bootstrap",
            "src.biored_cawv_evaluation",
            ("--split", "test", "--resamples", "1000", "--seed", "42"),
            (results / "test" / "cawv" / "cawv_evaluation.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded bootstrap",
            "src.biored_bootstrap",
            ("--resamples", "1000", "--seed", "42"),
            (results / "test" / "bootstrap_results.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded Cohen kappa",
            "src.biored_kappa",
            ("--split", "test"),
            (
                results / "test" / "cohen_kappa_results.json",
                results / "test" / "kappa_alignment_diagnostics.json",
            ),
            env,
        ),
        Stage(
            "BioRED overlap-excluded kappa diagnostics",
            "src.biored_kappa_diagnostic",
            ("--split", "test"),
            (
                results / "test" / "kappa_diagnostic" / "kappa_diagnostic.json",
                results / "test" / "kappa_diagnostic" / "kappa_model_summary.csv",
            ),
            env,
        ),
        Stage(
            "BioRED overlap-excluded error taxonomy",
            "src.biored_error_taxonomy",
            ("--split", "test", "--models", *MODEL_KEYS),
            (results / "test" / "error_taxonomy" / "taxonomy_summary.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded relaxed-overlap evaluation",
            "src.relaxed_overlap_evaluation",
            ("--dataset", "biored", "--split", "test"),
            (results / "test" / "relaxed_overlap" / "relaxed_overlap_metrics.csv",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded CAWV ablation and class-wise baseline",
            "src.cawv_ablation",
            ("--dataset", "biored", "--resamples", "1000", "--seed", "42"),
            (results / "test" / "cawv" / "ablation" / "cawv_ablation_results.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded method controls",
            "src.method_controls",
            ("--dataset", "biored", "--resamples", "1000", "--seed", "42"),
            (results / "test" / "method_controls" / "method_controls.json",),
            env,
        ),
        Stage(
            "BioRED overlap-excluded development-gold sensitivity",
            "src.development_gold_sensitivity",
            ("--dataset", "biored", "--seeds", "10"),
            (results / "test" / "development_gold_sensitivity" / "development_gold_sensitivity.json",),
            env,
        ),
    ]
    return items


def comparison_stage() -> Stage:
    return Stage(
        "BioRED official vs overlap-excluded comparison",
        "src.biored_conventional_compare",
        (),
        (ROOT / "results" / "biored_sensitivity_comparison" / "conventional_official_vs_overlap_excluded.json",),
    )


def run_pipeline(*, force: bool = False) -> Path:
    check_raw_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_biored_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log:
        print("BIORED STUDY", file=log)
        for stage in official_stages():
            run_stage(stage, log, force=force)
        for stage in overlap_sensitivity_stages():
            run_stage(stage, log, force=force)
        run_stage(comparison_stage(), log, force=True)
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
    print("\nBIORED COMPLETE.")
    print(f"Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete BioRED experiment.")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
