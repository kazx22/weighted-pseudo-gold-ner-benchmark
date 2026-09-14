"""Plot the BioRED development-only weighted-threshold sweep."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt

from src.biored_config import figures_dir, frozen_config_file, require_file, results_dir

PAPER_COLOURS = {
    "precision": "#2166ac",
    "recall": "#d6604d",
    "f1": "#4dac26",
}


def main() -> None:
    csv_path = require_file(results_dir("dev") / "threshold_selection.csv", "BioRED threshold sweep")
    config_path = require_file(frozen_config_file(), "BioRED frozen pseudo-gold configuration")

    import json
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = float(config["selected_threshold"])

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(row[key]) for key in ("threshold", "precision", "recall", "f1")})

    thresholds = [row["threshold"] for row in rows]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(thresholds, [row["precision"] for row in rows], label="Precision", color=PAPER_COLOURS["precision"], marker="o", markersize=3)
    axis.plot(thresholds, [row["recall"] for row in rows], label="Recall", color=PAPER_COLOURS["recall"], marker="o", markersize=3)
    axis.plot(thresholds, [row["f1"] for row in rows], label="F1", color=PAPER_COLOURS["f1"], marker="o", markersize=3)
    axis.axvline(selected, linestyle="--", linewidth=1.2, color="black", label=f"Selected = {selected:.4f}")
    axis.set_xlabel("Weighted-voting threshold")
    axis.set_ylabel("Exact-span score")
    axis.set_ylim(0, 1.05)
    axis.set_title("BioRED Development: Weighted-Threshold Selection")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = figures_dir("dev") / "biored_threshold_sensitivity_curve.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
