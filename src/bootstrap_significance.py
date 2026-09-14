"""Paired document-level bootstrap tests using exact entity-span counts.

This version evaluates both the five individual NER systems and the two
pseudo-gold references.  It keeps the original ten model-vs-model tests as one
Holm-corrected family and adds the paper's two central pseudo-gold comparisons
as a separate Holm-corrected family:

1. Weighted Pseudo-Gold vs Majority Pseudo-Gold
2. Weighted Pseudo-Gold vs scispaCy

No model inference is performed.  The script only reads the existing test
prediction and pseudo-gold JSONL files and resamples the 500 test documents.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from src.experiment_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    docs_file,
    gold_entities_file,
    normalize_split,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import group_by_row, load_jsonl, metrics_from_counts, per_document_exact_counts

N_BOOTSTRAP = 1000
SEED = 42

# Use the publication spelling in newly generated outputs.
DISPLAY_NAMES = {
    **MODEL_DISPLAY_NAMES,
    "scispacy": "scispaCy",
    "majority_pseudo_gold": "Majority Pseudo-Gold",
    "weighted_pseudo_gold": "Weighted Pseudo-Gold",
}


def aggregate_metrics(counts: list[dict], indices: np.ndarray) -> dict:
    """Aggregate exact-span TP/FP/FN over a sampled set of documents."""
    tp = sum(counts[int(index)]["tp"] for index in indices)
    fp = sum(counts[int(index)]["fp"] for index in indices)
    fn = sum(counts[int(index)]["fn"] for index in indices)
    return metrics_from_counts(tp, fp, fn)


def aggregate_f1(counts: list[dict], indices: np.ndarray) -> float:
    return float(aggregate_metrics(counts, indices)["f1"])


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    """Holm step-down family-wise error correction."""
    number = len(raw_p_values)
    if number == 0:
        return []

    order = sorted(range(number), key=lambda index: raw_p_values[index])
    adjusted = [0.0] * number
    running_max = 0.0

    for rank, original_index in enumerate(order):
        value = min(1.0, (number - rank) * raw_p_values[original_index])
        running_max = max(running_max, value)
        adjusted[original_index] = running_max

    return adjusted


def paired_bootstrap_p_value(
    first_bootstrap: np.ndarray,
    second_bootstrap: np.ndarray,
    *,
    resamples: int,
) -> float:
    """Two-sided paired-bootstrap sign-reversal probability.

    A +1 finite-sample correction prevents a reported p-value of exactly zero.
    """
    difference = first_bootstrap - second_bootstrap

    lower_tail = (np.count_nonzero(difference <= 0) + 1) / (resamples + 1)
    upper_tail = (np.count_nonzero(difference >= 0) + 1) / (resamples + 1)

    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


def load_system_counts(split: str, docs: list[dict], gold_by_row: dict) -> dict[str, list[dict]]:
    """Load existing predictions and produce per-document exact-span counts."""
    system_files: dict[str, Path] = {
        model_key: prediction_file(model_key, split) for model_key in MODEL_KEYS
    }
    system_files["majority_pseudo_gold"] = pseudo_gold_file("majority", split)
    system_files["weighted_pseudo_gold"] = pseudo_gold_file("weighted", split)

    per_system_counts: dict[str, list[dict]] = {}

    for system_key, path in system_files.items():
        predictions = load_jsonl(
            require_file(path, f"{DISPLAY_NAMES[system_key]} predictions")
        )
        per_system_counts[system_key] = per_document_exact_counts(
            docs,
            gold_by_row,
            group_by_row(predictions),
        )
        print(
            f"Loaded {DISPLAY_NAMES[system_key]:22} "
            f"from {path.name}"
        )

    return per_system_counts


def build_pairwise_rows(
    comparisons: list[tuple[str, str]],
    observed_f1: dict[str, float],
    bootstrap_f1: dict[str, np.ndarray],
    *,
    resamples: int,
) -> list[dict]:
    rows: list[dict] = []
    raw_p_values: list[float] = []

    for first, second in comparisons:
        observed_difference = observed_f1[first] - observed_f1[second]
        raw_p = paired_bootstrap_p_value(
            bootstrap_f1[first],
            bootstrap_f1[second],
            resamples=resamples,
        )
        raw_p_values.append(raw_p)
        rows.append(
            {
                "system_1": DISPLAY_NAMES[first],
                "system_2": DISPLAY_NAMES[second],
                "f1_system_1": observed_f1[first],
                "f1_system_2": observed_f1[second],
                "f1_difference": observed_difference,
                "raw_p": raw_p,
            }
        )

    adjusted_values = holm_adjust(raw_p_values)

    for row, adjusted_p in zip(rows, adjusted_values):
        row["holm_p"] = adjusted_p
        row["significant_raw_0.05"] = row["raw_p"] < 0.05
        row["significant_holm_0.05"] = adjusted_p < 0.05

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_pairwise_section(title: str, rows: list[dict]) -> None:
    print("\n" + title)
    print("=" * len(title))

    for row in rows:
        print(
            f"  {row['system_1']:22} vs {row['system_2']:22} "
            f"diff={row['f1_difference']:+.4f} "
            f"raw p={row['raw_p']:.4f} "
            f"Holm p={row['holm_p']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired exact-span bootstrap for BC5CDR models and pseudo-gold."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test", "development"],
    )
    parser.add_argument("--resamples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split = normalize_split(args.split, allow_train=False)
    if args.resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")

    docs = load_jsonl(require_file(docs_file(split), "parsed documents"))
    gold = load_jsonl(require_file(gold_entities_file(split), "human gold"))
    gold_by_row = group_by_row(gold)

    per_system_counts = load_system_counts(split, docs, gold_by_row)
    system_keys = list(MODEL_KEYS) + ["majority_pseudo_gold", "weighted_pseudo_gold"]

    n_docs = len(docs)
    full_indices = np.arange(n_docs)

    observed_metrics = {
        system_key: aggregate_metrics(per_system_counts[system_key], full_indices)
        for system_key in system_keys
    }
    observed_f1 = {
        system_key: float(observed_metrics[system_key]["f1"])
        for system_key in system_keys
    }

    print("\nObserved exact-span scores")
    print("==========================")
    for system_key in system_keys:
        metrics = observed_metrics[system_key]
        print(
            f"  {DISPLAY_NAMES[system_key]:22} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

    rng = np.random.default_rng(args.seed)
    bootstrap_f1 = {
        system_key: np.zeros(args.resamples, dtype=float)
        for system_key in system_keys
    }

    for iteration in range(args.resamples):
        sample = rng.integers(0, n_docs, size=n_docs)

        for system_key in system_keys:
            bootstrap_f1[system_key][iteration] = aggregate_f1(
                per_system_counts[system_key],
                sample,
            )

        if (iteration + 1) % 200 == 0:
            print(f"Completed {iteration + 1}/{args.resamples} resamples")

    confidence_intervals: list[dict] = []

    print("\n95% bootstrap confidence intervals")
    print("==================================")
    for system_key in system_keys:
        low, high = np.percentile(bootstrap_f1[system_key], [2.5, 97.5])
        row = {
            "system": DISPLAY_NAMES[system_key],
            "precision": float(observed_metrics[system_key]["precision"]),
            "recall": float(observed_metrics[system_key]["recall"]),
            "f1": observed_f1[system_key],
            "ci_low": float(low),
            "ci_high": float(high),
        }
        confidence_intervals.append(row)
        print(
            f"  {row['system']:22} F1={row['f1']:.4f} "
            f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )

    # Existing benchmark family: all ten comparisons among the five models.
    model_comparisons = list(itertools.combinations(MODEL_KEYS, 2))
    model_pairwise = build_pairwise_rows(
        model_comparisons,
        observed_f1,
        bootstrap_f1,
        resamples=args.resamples,
    )

    # Paper's central claims: treated as a separate two-comparison family.
    primary_comparisons = [
        ("weighted_pseudo_gold", "majority_pseudo_gold"),
        ("weighted_pseudo_gold", "scispacy"),
    ]
    primary_pairwise = build_pairwise_rows(
        primary_comparisons,
        observed_f1,
        bootstrap_f1,
        resamples=args.resamples,
    )

    print_pairwise_section(
        "Model-vs-model paired bootstrap tests (Holm over 10 comparisons)",
        model_pairwise,
    )
    print_pairwise_section(
        "Primary pseudo-gold paired bootstrap tests (Holm over 2 comparisons)",
        primary_pairwise,
    )

    output_dir = results_dir(split)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "split": split,
        "metric": "exact_character_span_and_label_micro_f1",
        "resamples": args.resamples,
        "seed": args.seed,
        "confidence_intervals": confidence_intervals,
        # Kept under the old key for compatibility with the existing manuscript workflow.
        "pairwise_tests": model_pairwise,
        "model_pairwise_tests": model_pairwise,
        "primary_pseudo_gold_tests": primary_pairwise,
        "multiple_testing_families": {
            "model_pairwise_tests": "Holm correction across 10 model comparisons",
            "primary_pseudo_gold_tests": "Holm correction across 2 prespecified central comparisons",
        },
    }

    save_json(output_payload, output_dir / "bootstrap_results.json")
    write_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        confidence_intervals,
    )
    write_csv(
        output_dir / "bootstrap_pairwise.csv",
        model_pairwise,
    )
    write_csv(
        output_dir / "bootstrap_primary_comparisons.csv",
        primary_pairwise,
    )

    print(f"\nSaved updated bootstrap results to {output_dir}")
    print("New central-comparison file:")
    print(f"  {output_dir / 'bootstrap_primary_comparisons.csv'}")


if __name__ == "__main__":
    main()
