"""Token-level BIO Cohen's kappa for a selected BC5CDR split.

Kappa is a secondary agreement analysis. Because the O label is frequent, it
must be interpreted alongside exact span precision, recall and F1.
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.experiment_config import (
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
    flat: list[str] = []
    for doc in docs:
        row_id = int(doc["row_id"])
        _, labels = span_to_bio(
            str(doc["full_text"]), entities_by_row.get(row_id, [])
        )
        flat.extend(labels)
    return flat


def interpret_kappa(value: float) -> str:
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


def plot_heatmap(names: list[str], matrix: np.ndarray, output_file) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for row in range(len(names)):
        for column in range(len(names)):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cohen's kappa", rotation=270, labelpad=18)
    ax.set_title("Pairwise Inter-Model Agreement\nToken-level BIO labels")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="BC5CDR Cohen's kappa analysis.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test", "development"],
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    docs = load_jsonl(require_file(docs_file(split), "parsed documents"))
    human_gold = load_jsonl(require_file(gold_entities_file(split), "human gold"))

    labels: dict[str, list[str]] = {
        "Human Gold": build_flat_labels(docs, group_by_row(human_gold))
    }
    for model_key in MODEL_KEYS:
        predictions = load_jsonl(
            require_file(prediction_file(model_key, split), f"{model_key} predictions")
        )
        labels[MODEL_DISPLAY_NAMES[model_key]] = build_flat_labels(
            docs, group_by_row(predictions)
        )

    lengths = {name: len(values) for name, values in labels.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Token-length mismatch: {lengths}")

    model_names = [MODEL_DISPLAY_NAMES[key] for key in MODEL_KEYS]
    matrix = np.eye(len(model_names), dtype=float)
    pairwise: list[dict] = []

    print("Pairwise inter-model Cohen's kappa:")
    for first_index, second_index in itertools.combinations(range(len(model_names)), 2):
        first = model_names[first_index]
        second = model_names[second_index]
        value = float(cohen_kappa_score(labels[first], labels[second]))
        matrix[first_index, second_index] = value
        matrix[second_index, first_index] = value
        pairwise.append(
            {
                "model_1": first,
                "model_2": second,
                "kappa": value,
                "interpretation": interpret_kappa(value),
            }
        )
        print(f"  {first:15} vs {second:15} k={value:.4f}")

    versus_gold: list[dict] = []
    print("\nModel versus human gold Cohen's kappa:")
    for model_name in model_names:
        value = float(cohen_kappa_score(labels[model_name], labels["Human Gold"]))
        versus_gold.append(
            {
                "model": model_name,
                "kappa": value,
                "interpretation": interpret_kappa(value),
            }
        )
        print(f"  {model_name:15} k={value:.4f} ({interpret_kappa(value)})")

    output = {
        "split": split,
        "level": "token-level BIO labels",
        "caution": "The frequent O class can inflate agreement; interpret with exact span metrics.",
        "pairwise": pairwise,
        "versus_human_gold": versus_gold,
        "mean_pairwise_kappa": float(np.mean([row["kappa"] for row in pairwise])),
    }
    save_json(output, results_dir(split) / "cohen_kappa_results.json")
    heatmap_path = figures_dir(split) / "kappa_heatmap.png"
    plot_heatmap(model_names, matrix, heatmap_path)
    print(f"\nSaved kappa results to {results_dir(split)}")
    print(f"Saved heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
