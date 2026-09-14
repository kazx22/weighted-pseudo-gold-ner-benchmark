"""Paired document-level bootstrap for the BioRED MedGemma extension."""

from __future__ import annotations

import argparse
import csv

import numpy as np

from src.biored_config import (
    docs_file,
    gold_entities_file,
    medgemma_hybrid_file,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import group_by_row, load_jsonl, metrics_from_counts, per_document_exact_counts

def holm_adjust(raw_p_values: list[float]) -> list[float]:
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
    difference = first_bootstrap - second_bootstrap
    lower_tail = (np.count_nonzero(difference <= 0) + 1) / (resamples + 1)
    upper_tail = (np.count_nonzero(difference >= 0) + 1) / (resamples + 1)
    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


DISPLAY = {
    "weighted": "Weighted Pseudo-Gold",
    "scispacy": "scispaCy",
    "medgemma": "MedGemma Zero-Shot",
    "hybrid": "Weighted + MedGemma Tie-Breaker",
}


def aggregate(counts: list[dict], indices: np.ndarray) -> dict:
    tp = fp = fn = 0
    for index in indices:
        item = counts[int(index)]
        tp += int(item["tp"])
        fp += int(item["fp"])
        fn += int(item["fn"])
    return metrics_from_counts(tp, fp, fn)


def write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap BioRED MedGemma comparisons.")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = load_jsonl(require_file(docs_file("test"), "BioRED test documents"))
    gold = load_jsonl(require_file(gold_entities_file("test"), "BioRED test gold"))
    gold_by_row = group_by_row(gold)
    paths = {
        "weighted": pseudo_gold_file("weighted", "test"),
        "scispacy": prediction_file("scispacy", "test"),
        "medgemma": prediction_file("medgemma", "test"),
        "hybrid": medgemma_hybrid_file("test"),
    }

    counts: dict[str, list[dict]] = {}
    for key, path in paths.items():
        predictions = load_jsonl(require_file(path, f"{DISPLAY[key]} predictions"))
        counts[key] = per_document_exact_counts(docs, gold_by_row, group_by_row(predictions))

    full = np.arange(len(docs))
    observed = {key: aggregate(value, full) for key, value in counts.items()}
    rng = np.random.default_rng(args.seed)
    samples = {key: np.zeros(args.resamples, dtype=float) for key in counts}
    for iteration in range(args.resamples):
        indices = rng.integers(0, len(docs), size=len(docs))
        for key in counts:
            samples[key][iteration] = float(aggregate(counts[key], indices)["f1"])
        if (iteration + 1) % 200 == 0:
            print(f"Completed {iteration + 1}/{args.resamples} resamples")

    confidence_intervals: list[dict] = []
    for key in counts:
        low, high = np.percentile(samples[key], [2.5, 97.5])
        confidence_intervals.append(
            {
                "system": DISPLAY[key],
                "precision": observed[key]["precision"],
                "recall": observed[key]["recall"],
                "f1": observed[key]["f1"],
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    comparisons = [("hybrid", "weighted"), ("medgemma", "scispacy"), ("hybrid", "medgemma")]
    rows: list[dict] = []
    raw_values: list[float] = []
    for first, second in comparisons:
        raw_p = paired_bootstrap_p_value(samples[first], samples[second], resamples=args.resamples)
        raw_values.append(raw_p)
        rows.append(
            {
                "system_1": DISPLAY[first],
                "system_2": DISPLAY[second],
                "f1_system_1": observed[first]["f1"],
                "f1_system_2": observed[second]["f1"],
                "f1_difference": observed[first]["f1"] - observed[second]["f1"],
                "raw_p": raw_p,
            }
        )

    for row, holm_p in zip(rows, holm_adjust(raw_values)):
        row["holm_p"] = holm_p
        row["significant_holm_0.05"] = holm_p < 0.05

    output_dir = results_dir("test")
    save_json(
        {
            "dataset": "BioRED",
            "task_scope": ["DISEASE", "CHEMICAL"],
            "split": "test",
            "metric": "exact_character_span_and_label_micro_f1",
            "resamples": args.resamples,
            "seed": args.seed,
            "confidence_intervals": confidence_intervals,
            "comparisons": rows,
            "multiple_testing_family": "Holm correction across 3 BioRED MedGemma comparisons",
        },
        output_dir / "medgemma_bootstrap_results.json",
    )
    write_csv(output_dir / "medgemma_bootstrap_confidence_intervals.csv", confidence_intervals)
    write_csv(output_dir / "medgemma_bootstrap_comparisons.csv", rows)

    for row in rows:
        print(
            f"{row['system_1']} vs {row['system_2']}: diff={row['f1_difference']:+.4f}, "
            f"raw p={row['raw_p']:.4f}, Holm p={row['holm_p']:.4f}"
        )


if __name__ == "__main__":
    main()
