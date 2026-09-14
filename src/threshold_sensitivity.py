"""Plot development-set threshold selection produced by candidate_gold.py."""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.experiment_config import figures_dir, require_file, results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frozen threshold selection.")
    parser.parse_args()

    csv_path = require_file(
        results_dir("dev") / "threshold_selection.csv",
        "development threshold-selection table",
    )

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "threshold": float(row["threshold"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                }
            )

    if not rows:
        raise ValueError(f"No threshold rows found in {csv_path}")

    thresholds = [row["threshold"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    f1 = [row["f1"] for row in rows]
    best_index = max(
        range(len(rows)),
        key=lambda index: (f1[index], precision[index], thresholds[index]),
    )

    output_dir = figures_dir("dev")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "threshold_sensitivity_curve.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precision, marker="o", label="Precision")
    ax.plot(thresholds, recall, marker="s", label="Recall")
    ax.plot(thresholds, f1, marker="^", label="F1-score")
    ax.axvline(thresholds[best_index], linestyle="--", linewidth=1.2)
    ax.annotate(
        f"Selected threshold = {thresholds[best_index]:.4f}\n"
        f"Development F1 = {f1[best_index]:.4f}",
        xy=(thresholds[best_index], f1[best_index]),
        xytext=(10, -50),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xlabel("Weighted-voting threshold")
    ax.set_ylabel("Exact-span score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Development-Set Pseudo-Gold Threshold Selection")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected threshold: {thresholds[best_index]:.8f}")
    print(f"Development F1:     {f1[best_index]:.4f}")
    print(f"Saved figure to {output_file}")


if __name__ == "__main__":
    main()
