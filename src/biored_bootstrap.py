"""Document-level bootstrap validation for BioRED disease/chemical NER."""

from __future__ import annotations

import argparse
import csv
import itertools

import numpy as np

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    docs_file,
    gold_entities_file,
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



def aggregate(counts: list[dict], indices: np.ndarray) -> dict:
    tp = fp = fn = 0
    for index in indices:
        item = counts[int(index)]
        tp += int(item["tp"])
        fp += int(item["fp"])
        fn += int(item["fn"])
    return metrics_from_counts(tp, fp, fn)


def write_csv(path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap BioRED test results.")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = load_jsonl(require_file(docs_file("test"), "BioRED test documents"))
    gold = load_jsonl(require_file(gold_entities_file("test"), "BioRED test gold"))
    gold_by_row = group_by_row(gold)

    paths = {
        "majority": pseudo_gold_file("majority", "test"),
        "weighted": pseudo_gold_file("weighted", "test"),
    }
    display = {
        "majority": "Majority Pseudo-Gold",
        "weighted": "Weighted Pseudo-Gold",
    }
    for model_key in MODEL_KEYS:
        paths[model_key] = prediction_file(model_key, "test")
        display[model_key] = MODEL_DISPLAY_NAMES[model_key]

    counts: dict[str, list[dict]] = {}
    for key, path in paths.items():
        predictions = load_jsonl(require_file(path, f"{display[key]} predictions"))
        counts[key] = per_document_exact_counts(docs, gold_by_row, group_by_row(predictions))

    full_indices = np.arange(len(docs))
    observed = {key: aggregate(value, full_indices) for key, value in counts.items()}
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
                "system": display[key],
                "precision": observed[key]["precision"],
                "recall": observed[key]["recall"],
                "f1": observed[key]["f1"],
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    primary_pairs = [("weighted", "majority"), ("weighted", "scispacy")]
    primary_rows: list[dict] = []
    primary_raw: list[float] = []
    for first, second in primary_pairs:
        raw_p = paired_bootstrap_p_value(samples[first], samples[second], resamples=args.resamples)
        primary_raw.append(raw_p)
        primary_rows.append(
            {
                "system_1": display[first],
                "system_2": display[second],
                "f1_difference": float(observed[first]["f1"] - observed[second]["f1"]),
                "raw_p": raw_p,
            }
        )
    for row, adjusted in zip(primary_rows, holm_adjust(primary_raw)):
        row["holm_p"] = adjusted
        row["significant_holm_0.05"] = adjusted < 0.05

    pairwise_rows: list[dict] = []
    pairwise_raw: list[float] = []
    for first, second in itertools.combinations(MODEL_KEYS, 2):
        raw_p = paired_bootstrap_p_value(samples[first], samples[second], resamples=args.resamples)
        pairwise_raw.append(raw_p)
        pairwise_rows.append(
            {
                "model_1": display[first],
                "model_2": display[second],
                "f1_difference": float(observed[first]["f1"] - observed[second]["f1"]),
                "raw_p": raw_p,
            }
        )
    for row, adjusted in zip(pairwise_rows, holm_adjust(pairwise_raw)):
        row["holm_p"] = adjusted
        row["significant_holm_0.05"] = adjusted < 0.05

    output_dir = results_dir("test")
    payload = {
        "dataset": "BioRED",
        "task_scope": ["DISEASE", "CHEMICAL"],
        "split": "test",
        "resamples": args.resamples,
        "seed": args.seed,
        "metric": "exact_character_span_and_label_micro_f1",
        "confidence_intervals": confidence_intervals,
        "primary_comparisons": primary_rows,
        "model_pairwise_comparisons": pairwise_rows,
    }
    save_json(payload, output_dir / "bootstrap_results.json")
    write_csv(output_dir / "bootstrap_confidence_intervals.csv", confidence_intervals)
    write_csv(output_dir / "bootstrap_primary_comparisons.csv", primary_rows)
    write_csv(output_dir / "bootstrap_pairwise.csv", pairwise_rows)

    for row in primary_rows:
        print(
            f"{row['system_1']} vs {row['system_2']}: "
            f"diff={row['f1_difference']:+.4f}, raw p={row['raw_p']:.4f}, "
            f"Holm p={row['holm_p']:.4f}"
        )


if __name__ == "__main__":
    main()
