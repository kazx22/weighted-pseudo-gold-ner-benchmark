"""Evaluate BioRED MedGemma zero-shot and selective-hybrid outputs."""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from src.biored_config import (
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

PAPER_COLOURS = {
    "precision": "#2166ac",
    "recall": "#d6604d",
    "f1": "#4dac26",
    "disease": "#b2182b",
    "chemical": "#2166ac",
}


def evaluate(name: str, gold: list[dict], predictions: list[dict]) -> dict:
    metrics = exact_span_metrics(gold, predictions)
    return {
        "system": name,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "per_label": exact_per_label_report(gold, predictions),
    }


def write_csv(path, rows: list[dict]) -> None:
    fields = ["system", "precision", "recall", "f1", "tp", "fp", "fn"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot_results(path, rows: list[dict], split: str) -> None:
    names = [row["system"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 6.2))
    bars = [
        axis.bar(x - width, [row["precision"] for row in rows], width, label="Precision", color=PAPER_COLOURS["precision"]),
        axis.bar(x, [row["recall"] for row in rows], width, label="Recall", color=PAPER_COLOURS["recall"]),
        axis.bar(x + width, [row["f1"] for row in rows], width, label="F1", color=PAPER_COLOURS["f1"]),
    ]
    for group in bars:
        for bar in group:
            axis.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=18, ha="right")
    axis.set_ylim(0, 1.06)
    axis.set_ylabel("Exact-span score")
    axis.set_title(f"BioRED {split}: MedGemma Comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioRED MedGemma outputs.")
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    args = parser.parse_args()
    split = normalize_split(args.split)

    gold = load_jsonl(require_file(gold_entities_file(split), "BioRED human gold"))
    systems = [
        ("Majority Pseudo-Gold", pseudo_gold_file("majority", split)),
        ("Weighted Pseudo-Gold", pseudo_gold_file("weighted", split)),
        ("scispaCy", prediction_file("scispacy", split)),
        ("MedGemma Zero-Shot", prediction_file("medgemma", split)),
        ("Weighted + MedGemma Tie-Breaker", medgemma_hybrid_file(split)),
    ]

    rows: list[dict] = []
    for name, path in systems:
        predictions = load_jsonl(require_file(path, f"{name} predictions"))
        row = evaluate(name, gold, predictions)
        rows.append(row)
        print(f"{name:34} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")

    zero_summary_path = medgemma_results_dir(split) / "zero_shot_summary.json"
    hybrid_summary_path = medgemma_hybrid_results_dir(split) / "hybrid_summary.json"
    zero_summary = json.loads(zero_summary_path.read_text(encoding="utf-8")) if zero_summary_path.exists() else None
    hybrid_summary = json.loads(hybrid_summary_path.read_text(encoding="utf-8")) if hybrid_summary_path.exists() else None

    output_dir = results_dir(split)
    save_json(
        {
            "dataset": "BioRED",
            "task_scope": ["DISEASE", "CHEMICAL"],
            "split": split,
            "primary_metric": "exact_character_span_and_label",
            "results": rows,
            "zero_shot_diagnostics": zero_summary,
            "hybrid_routing": hybrid_summary,
        },
        output_dir / "medgemma_evaluation.json",
    )
    write_csv(output_dir / "medgemma_evaluation.csv", rows)
    plot_results(output_dir / "figures" / "biored_medgemma_comparison.png", rows, split)


if __name__ == "__main__":
    main()
