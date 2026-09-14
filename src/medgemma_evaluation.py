"""Evaluate MedGemma zero-shot and selective-hybrid BC5CDR outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.experiment_config import (
    gold_entities_file,
    medgemma_hybrid_file,
    medgemma_hybrid_results_dir,
    medgemma_results_dir,
    normalize_split,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import exact_per_label_report, exact_span_metrics, load_jsonl


def evaluate(name: str, gold: list[dict], predictions: list[dict]) -> dict:
    metrics = exact_span_metrics(gold, predictions)
    return {
        "system": name,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "per_label": exact_per_label_report(gold, predictions),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["system", "precision", "recall", "f1", "tp", "fp", "fn"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot_results(path: Path, rows: list[dict], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["system"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    f1 = [row["f1"] for row in rows]
    x = list(range(len(rows)))
    width = 0.25

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar([value - width for value in x], precision, width, label="Precision")
    axis.bar(x, recall, width, label="Recall")
    axis.bar([value + width for value in x], f1, width, label="F1")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Exact-span score")
    axis.set_title(f"MedGemma BC5CDR comparison — {split}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedGemma BC5CDR outputs.")
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    gold = load_jsonl(require_file(gold_entities_file(split), "human gold"))
    systems = [
        (
            "Majority Pseudo-Gold",
            pseudo_gold_file("majority", split),
        ),
        (
            "Weighted Pseudo-Gold",
            pseudo_gold_file("weighted", split),
        ),
        (
            "scispaCy",
            prediction_file("scispacy", split),
        ),
        (
            "MedGemma Zero-Shot",
            prediction_file("medgemma", split),
        ),
        (
            "Weighted + MedGemma Tie-Breaker",
            medgemma_hybrid_file(split),
        ),
    ]

    rows: list[dict] = []
    for name, path in systems:
        predictions = load_jsonl(require_file(path, f"{name} predictions"))
        row = evaluate(name, gold, predictions)
        rows.append(row)
        print(
            f"{name:34} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f}"
        )

    output_dir = results_dir(split)
    zero_summary_path = medgemma_results_dir(split) / "zero_shot_summary.json"
    hybrid_summary_path = medgemma_hybrid_results_dir(split) / "hybrid_summary.json"
    zero_summary = None
    hybrid_summary = None
    if zero_summary_path.exists():
        zero_summary = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    if hybrid_summary_path.exists():
        hybrid_summary = json.loads(hybrid_summary_path.read_text(encoding="utf-8"))

    payload = {
        "split": split,
        "primary_metric": "exact_character_span_and_label",
        "results": rows,
        "zero_shot_diagnostics": zero_summary,
        "hybrid_routing": hybrid_summary,
    }
    save_json(payload, output_dir / "medgemma_evaluation.json")
    write_csv(output_dir / "medgemma_evaluation.csv", rows)
    plot_results(output_dir / "figures" / "medgemma_comparison.png", rows, split)
    print(f"Saved MedGemma evaluation to {output_dir}")


if __name__ == "__main__":
    main()
