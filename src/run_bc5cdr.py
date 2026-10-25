from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.pipeline_common import MODEL_KEYS, ROOT, Stage, check_raw_inputs, run_stage

from src.symmetric_reporting import generate as generate_symmetric_reporting

LOG_DIR = ROOT / "logs"


def stages() -> list[Stage]:
    proc = ROOT / "data" / "processed" / "bc5cdr"
    gold = ROOT / "data" / "gold"
    dev = ROOT / "results" / "dev"
    test = ROOT / "results" / "test"
    model_modules = {
        "scispacy": "src.scispacy_bc5cdr",
        "biobert": "src.biobert_bc5cdr",
        "pubmedbert": "src.pubmed_bc5cdr",
        "clinicalbert": "src.clinicalbert_bc5cdr",
        "d4data": "src.d4data_bc5cdr",
    }

    items: list[Stage] = [
        Stage(
            "BC5CDR parse corpus",
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
        items.append(
            Stage(
                f"BC5CDR development model: {model}",
                model_modules[model],
                ("--split", "dev"),
                (proc / f"{model}_dev_entities_bc5cdr.jsonl",),
            )
        )

    items.extend(
        [
            Stage(
                "BC5CDR validate development predictions",
                "src.span_integrity_audit",
                ("--dataset", "bc5cdr", "--split", "dev", "--top", "50", "--fail-on-integrity-error"),
                (dev / "span_audit" / "span_integrity_audit.json",),
            ),
            Stage(
                "BC5CDR fit weighted voting on development",
                "src.candidate_gold",
                ("--split", "dev"),
                (dev / "frozen_pseudo_gold_config.json",),
            ),
            Stage(
                "BC5CDR weighted-threshold sensitivity",
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
        items.append(
            Stage(
                f"BC5CDR test model: {model}",
                model_modules[model],
                ("--split", "test"),
                (proc / f"{model}_test_entities_bc5cdr.jsonl",),
                resource_required=True,
            )
        )

    items.extend(
        [
            Stage(
                "BC5CDR validate test predictions",
                "src.span_integrity_audit",
                ("--dataset", "bc5cdr", "--split", "test", "--top", "50", "--fail-on-integrity-error"),
                (test / "span_audit" / "span_integrity_audit.json",),
            ),
            Stage(
                "BC5CDR apply frozen weighted voting to test",
                "src.candidate_gold",
                ("--split", "test"),
                (test / "pseudo_gold_application.json",),
            ),
            Stage(
                "BC5CDR test evaluation",
                "src.bc5cdr_evaluation",
                ("--split", "test"),
                (test / "evaluation_results.json",),
            ),
            Stage(
                "BC5CDR weighted-voting bootstrap",
                "src.bootstrap_significance",
                ("--split", "test", "--resamples", "1000", "--seed", "42"),
                (test / "bootstrap_results.json",),
            ),
            Stage(
                "BC5CDR Cohen kappa + positive-token diagnostics",
                "src.cohen_kappa",
                ("--split", "test"),
                (
                    test / "cohen_kappa_results.json",
                    test / "kappa_diagnostic" / "kappa_diagnostic.json",
                    test / "kappa_diagnostic" / "kappa_model_summary.csv",
                ),
            ),
            Stage(
                "BC5CDR error taxonomy",
                "src.error_taxonomy",
                ("--split", "test", "--models", *MODEL_KEYS),
                (
                    test / "error_taxonomy" / "scispacy_error_counts.txt",
                    test / "error_taxonomy" / "biobert_error_counts.txt",
                    test / "error_taxonomy" / "pubmedbert_error_counts.txt",
                    test / "error_taxonomy" / "clinicalbert_error_counts.txt",
                    test / "error_taxonomy" / "d4data_error_counts.txt",
                    test / "error_taxonomy" / "taxonomy_summary.json",
                ),
            ),
            Stage(
                "BC5CDR relaxed-overlap evaluation",
                "src.relaxed_overlap_evaluation",
                ("--dataset", "bc5cdr", "--split", "test"),
                (test / "relaxed_overlap" / "relaxed_overlap_metrics.csv",),
            ),
            Stage(
                "BC5CDR tuned-unweighted and leave-one-model-out controls",
                "src.method_controls",
                ("--dataset", "bc5cdr", "--resamples", "1000", "--seed", "42"),
                (test / "method_controls" / "method_controls.json",),
            ),
            Stage(
                "BC5CDR development-gold sensitivity",
                "src.development_gold_sensitivity",
                ("--dataset", "bc5cdr", "--seeds", "10"),
                (test / "development_gold_sensitivity" / "development_gold_sensitivity.json",),
            ),
            Stage(
                "BC5CDR CAWV fit on development",
                "src.content_aware_voting",
                ("--split", "dev"),
                (
                    dev / "cawv" / "frozen_cawv_config.json",
                    dev / "figures" / "cawv_alpha_sensitivity.png",
                    dev / "figures" / "cawv_threshold_curve_selected_alpha.png",
                ),
            ),
            Stage(
                "BC5CDR CAWV apply frozen settings to test",
                "src.content_aware_voting",
                ("--split", "test"),
                (gold / "cawv_pseudo_gold_test_entities_bc5cdr.jsonl",),
            ),
            Stage(
                "BC5CDR CAWV evaluation and bootstrap",
                "src.cawv_evaluation",
                ("--split", "test", "--resamples", "1000", "--seed", "42"),
                (test / "cawv" / "cawv_evaluation.json",),
            ),
            Stage(
                "BC5CDR CAWV ablation and class-wise baseline",
                "src.cawv_ablation",
                ("--dataset", "bc5cdr", "--resamples", "1000", "--seed", "42"),
                (test / "cawv" / "ablation" / "cawv_ablation_results.json",),
            ),
            Stage(
                "Refresh manuscript results transfer log",
                "src.manuscript_results_log",
                (),
                (ROOT / "results" / "MANUSCRIPT_RESULTS.log",),
                always_run=True,
            ),
        ]
    )
    return items


def run_pipeline(*, force: bool = False) -> Path:
    check_raw_inputs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_bc5cdr_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log:
        print("BC5CDR STUDY", file=log)
        for stage in stages():
            run_stage(stage, log, force=force)
    generate_symmetric_reporting(allow_partial=True)
    print("\nBC5CDR COMPLETE.")
    print(f"Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete BC5CDR experiment.")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
