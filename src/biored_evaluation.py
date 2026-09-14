"""Exact-span BioRED evaluation and paper-style figures for disease/chemical NER."""

from __future__ import annotations

import argparse
import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    figures_dir,
    gold_entities_file,
    load_json,
    normalize_split,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    runtime_file,
    save_json,
)
from src.utils import exact_per_label_report, exact_span_metrics, load_jsonl

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

PAPER_COLOURS = {
    "precision": "#2166ac",
    "recall": "#d6604d",
    "f1": "#4dac26",
    "disease": "#b2182b",
    "chemical": "#2166ac",
}


def evaluate(name: str, gold: list[dict], predictions: list[dict], runtime=None) -> dict:
    metrics = exact_span_metrics(gold, predictions)
    return {
        "system": name,
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "runtime_seconds_per_document": runtime,
        "per_label": exact_per_label_report(gold, predictions),
    }


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "runtime_seconds_per_document",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _annotate(axis, bars, decimals=3) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped(rows: list[dict], output_file, title: str) -> None:
    names = [row["system"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    p = axis.bar(
        x - width,
        [row["precision"] for row in rows],
        width,
        label="Precision",
        color=PAPER_COLOURS["precision"],
        edgecolor="white",
        linewidth=0.5,
    )
    r = axis.bar(
        x,
        [row["recall"] for row in rows],
        width,
        label="Recall",
        color=PAPER_COLOURS["recall"],
        edgecolor="white",
        linewidth=0.5,
    )
    f = axis.bar(
        x + width,
        [row["f1"] for row in rows],
        width,
        label="F1",
        color=PAPER_COLOURS["f1"],
        edgecolor="white",
        linewidth=0.5,
    )
    _annotate(axis, p)
    _annotate(axis, r)
    _annotate(axis, f)
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Exact-span score")
    axis.set_title(title)
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.grid(True, linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file)
    plt.close(figure)


def plot_per_label(rows: list[dict], output_file, title: str) -> None:
    x = np.arange(len(rows))
    width = 0.36
    disease = [float(row["per_label"]["DISEASE"]["f1-score"]) for row in rows]
    chemical = [float(row["per_label"]["CHEMICAL"]["f1-score"]) for row in rows]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    bars_d = axis.bar(
        x - width / 2,
        disease,
        width,
        label="DISEASE",
        color=PAPER_COLOURS["disease"],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_c = axis.bar(
        x + width / 2,
        chemical,
        width,
        label="CHEMICAL",
        color=PAPER_COLOURS["chemical"],
        edgecolor="white",
        linewidth=0.5,
    )
    _annotate(axis, bars_d)
    _annotate(axis, bars_c)
    axis.set_xticks(x)
    axis.set_xticklabels([row["system"] for row in rows], rotation=15, ha="right")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Exact-span F1")
    axis.set_title(title)
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.grid(True, linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioRED disease/chemical outputs.")
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    args = parser.parse_args()
    split = normalize_split(args.split)

    gold = load_jsonl(require_file(gold_entities_file(split), "BioRED human gold"))
    runtimes = load_json(runtime_file(split), default={}) or {}

    pseudo_rows: list[dict] = []
    for name, kind in (
        ("Majority Pseudo-Gold", "majority"),
        ("Weighted Pseudo-Gold", "weighted"),
    ):
        predictions = load_jsonl(require_file(pseudo_gold_file(kind, split), name))
        pseudo_rows.append(evaluate(name, gold, predictions))

    model_rows: list[dict] = []
    for model_key in MODEL_KEYS:
        predictions = load_jsonl(
            require_file(prediction_file(model_key, split), f"{MODEL_DISPLAY_NAMES[model_key]} predictions")
        )
        runtime = (runtimes.get(model_key) or {}).get("average_seconds_per_document")
        model_rows.append(evaluate(MODEL_DISPLAY_NAMES[model_key], gold, predictions, runtime))

    for row in pseudo_rows + model_rows:
        print(
            f"{row['system']:24} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f}"
        )

    payload = {
        "dataset": "BioRED",
        "task_scope": ["DISEASE", "CHEMICAL"],
        "split": split,
        "primary_metric": "exact_character_span_and_label",
        "pseudo_gold_vs_human": pseudo_rows,
        "models_vs_human": model_rows,
    }
    output_dir = results_dir(split)
    save_json(payload, output_dir / "evaluation_results.json")
    _write_csv(output_dir / "evaluation_summary.csv", pseudo_rows + model_rows)

    figure_dir = figures_dir(split)
    plot_grouped(
        pseudo_rows,
        figure_dir / "biored_pseudo_gold_comparison.png",
        f"BioRED {split}: Pseudo-Gold Against Human Gold",
    )
    plot_grouped(
        model_rows,
        figure_dir / "biored_models_vs_human_gold.png",
        f"BioRED {split}: Off-the-Shelf Models Against Human Gold",
    )
    plot_per_label(
        model_rows,
        figure_dir / "biored_per_label_f1.png",
        f"BioRED {split}: F1 by Entity Label",
    )
    print(f"Saved evaluation -> {output_dir}")
    print(f"Saved figures -> {figure_dir}")


if __name__ == "__main__":
    main()
