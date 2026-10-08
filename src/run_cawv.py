from __future__ import annotations

import argparse
import time

from src.pipeline_common import ROOT, Stage, biored_env, run_stage


def stages() -> list[Stage]:
    return [
        Stage(
            "BC5CDR development prediction span-integrity audit",
            "src.span_integrity_audit",
            ("--dataset", "bc5cdr", "--split", "dev", "--top", "50", "--fail-on-integrity-error"),
            (ROOT / "results" / "dev" / "span_audit" / "span_integrity_audit.json",),
        ),
        Stage(
            "BC5CDR test prediction span-integrity audit",
            "src.span_integrity_audit",
            ("--dataset", "bc5cdr", "--split", "test", "--top", "50", "--fail-on-integrity-error"),
            (ROOT / "results" / "test" / "span_audit" / "span_integrity_audit.json",),
        ),
        Stage(
            "BioRED official development prediction span-integrity audit",
            "src.span_integrity_audit",
            ("--dataset", "biored", "--split", "dev", "--top", "50", "--fail-on-integrity-error"),
            (ROOT / "results" / "biored" / "dev" / "span_audit" / "span_integrity_audit.json",),
            biored_env("official"),
        ),
        Stage(
            "BioRED official test prediction span-integrity audit",
            "src.span_integrity_audit",
            ("--dataset", "biored", "--split", "test", "--top", "50", "--fail-on-integrity-error"),
            (ROOT / "results" / "biored" / "test" / "span_audit" / "span_integrity_audit.json",),
            biored_env("official"),
        ),
        Stage(
            "BC5CDR CAWV fit on development",
            "src.content_aware_voting",
            ("--split", "dev"),
            (
                ROOT / "results" / "dev" / "cawv" / "frozen_cawv_config.json",
                ROOT / "results" / "dev" / "figures" / "cawv_alpha_sensitivity.png",
                ROOT / "results" / "dev" / "figures" / "cawv_threshold_curve_selected_alpha.png",
            ),
        ),
        Stage(
            "BC5CDR CAWV apply frozen config to test",
            "src.content_aware_voting",
            ("--split", "test"),
            (ROOT / "data" / "gold" / "cawv_pseudo_gold_test_entities_bc5cdr.jsonl",),
        ),
        Stage(
            "BC5CDR CAWV evaluation and bootstrap",
            "src.cawv_evaluation",
            ("--split", "test", "--resamples", "1000", "--seed", "42"),
            (ROOT / "results" / "test" / "cawv" / "cawv_evaluation.json",),
        ),
        Stage(
            "BC5CDR CAWV ablation and class-wise baseline",
            "src.cawv_ablation",
            ("--dataset", "bc5cdr", "--resamples", "1000", "--seed", "42"),
            (ROOT / "results" / "test" / "cawv" / "ablation" / "cawv_ablation_results.json",),
        ),
        Stage(
            "BioRED official CAWV fit on development",
            "src.biored_content_aware_voting",
            ("--split", "dev"),
            (
                ROOT / "results" / "biored" / "dev" / "cawv" / "frozen_cawv_config.json",
                ROOT / "results" / "biored" / "dev" / "figures" / "cawv_alpha_sensitivity.png",
                ROOT / "results" / "biored" / "dev" / "figures" / "cawv_threshold_curve_selected_alpha.png",
            ),
            biored_env("official"),
        ),
        Stage(
            "BioRED official CAWV apply frozen config to test",
            "src.biored_content_aware_voting",
            ("--split", "test"),
            (ROOT / "data" / "gold" / "biored" / "cawv_pseudo_gold_test_entities_biored.jsonl",),
            biored_env("official"),
        ),
        Stage(
            "BioRED official CAWV evaluation and bootstrap",
            "src.biored_cawv_evaluation",
            ("--split", "test", "--resamples", "1000", "--seed", "42"),
            (ROOT / "results" / "biored" / "test" / "cawv" / "cawv_evaluation.json",),
            biored_env("official"),
        ),
        Stage(
            "BioRED official CAWV ablation and class-wise baseline",
            "src.cawv_ablation",
            ("--dataset", "biored", "--resamples", "1000", "--seed", "42"),
            (ROOT / "results" / "biored" / "test" / "cawv" / "ablation" / "cawv_ablation_results.json",),
            biored_env("official"),
        ),
    ]


def run_pipeline(*, force: bool = False):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_cawv_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log:
        for stage in stages():
            run_stage(stage, log, force=force)
    print("\nCAWV PIPELINE COMPLETED.")
    print(f"Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
