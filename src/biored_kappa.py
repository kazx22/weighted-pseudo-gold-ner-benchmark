"""Token-level BIO Cohen's kappa diagnostics for BioRED disease/chemical NER."""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    docs_file,
    figures_dir,
    gold_entities_file,
    normalize_split,
    prediction_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import group_by_row, load_jsonl, span_to_bio


def build_flat_labels(docs: list[dict], entities_by_row: dict[int, list[dict]]) -> list[str]:
    output: list[str] = []
    for document in docs:
        row_id = int(document["row_id"])
        _, labels = span_to_bio(str(document["full_text"]), entities_by_row.get(row_id, []))
        output.extend(labels)
    return output


def interpret(value: float) -> str:
    if value < 0:
        return "Poor"
    if value < 0.20:
        return "Slight"
    if value < 0.40:
        return "Fair"
    if value < 0.60:
        return "Moderate"
    if value < 0.80:
        return "Substantial"
    return "Almost perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description="BioRED token-level kappa diagnostic.")
    parser.add_argument("--split", default="test", choices=["dev", "test", "development"])
    args = parser.parse_args()
    split = normalize_split(args.split)

    docs = load_jsonl(require_file(docs_file(split), "BioRED documents"))
    gold = load_jsonl(require_file(gold_entities_file(split), "BioRED human gold"))
    labels = {"Human Gold": build_flat_labels(docs, group_by_row(gold))}
    for key in MODEL_KEYS:
        predictions = load_jsonl(require_file(prediction_file(key, split), f"{key} predictions"))
        labels[MODEL_DISPLAY_NAMES[key]] = build_flat_labels(docs, group_by_row(predictions))

    model_names = [MODEL_DISPLAY_NAMES[key] for key in MODEL_KEYS]
    matrix = np.eye(len(model_names), dtype=float)
    pairwise: list[dict] = []
    for first_index, second_index in itertools.combinations(range(len(model_names)), 2):
        first = model_names[first_index]
        second = model_names[second_index]
        value = float(cohen_kappa_score(labels[first], labels[second]))
        matrix[first_index, second_index] = value
        matrix[second_index, first_index] = value
        pairwise.append(
            {"model_1": first, "model_2": second, "kappa": value, "interpretation": interpret(value)}
        )

    versus_gold: list[dict] = []
    for name in model_names:
        value = float(cohen_kappa_score(labels[name], labels["Human Gold"]))
        versus_gold.append({"model": name, "kappa": value, "interpretation": interpret(value)})
        print(f"{name:15} vs human gold: kappa={value:.4f}")

    save_json(
        {
            "dataset": "BioRED",
            "task_scope": ["DISEASE", "CHEMICAL"],
            "split": split,
            "level": "token-level BIO labels",
            "caution": "The frequent O class can inflate agreement; primary evaluation is exact span and label.",
            "pairwise": pairwise,
            "versus_human_gold": versus_gold,
            "mean_pairwise_kappa": float(np.mean([row["kappa"] for row in pairwise])),
        },
        results_dir(split) / "cohen_kappa_results.json",
    )

    fig, axis = plt.subplots(figsize=(8, 6.5))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(model_names)))
    axis.set_yticks(np.arange(len(model_names)))
    axis.set_xticklabels(model_names, rotation=45, ha="right")
    axis.set_yticklabels(model_names)
    for row in range(len(model_names)):
        for column in range(len(model_names)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Cohen's kappa", rotation=270, labelpad=18)
    axis.set_title(f"BioRED {split}: Pairwise Token-Level BIO Agreement")
    fig.tight_layout()
    output_file = figures_dir(split) / "biored_kappa_heatmap.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
