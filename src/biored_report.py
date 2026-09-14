"""Generate BioRED paper-ready figures and a compact result report from saved outputs.

No Ollama calls are made here. This is safe to rerun after the expensive LLM step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.biored_config import (
    docs_file,
    figures_dir,
    gold_entities_file,
    medgemma_hybrid_file,
    medgemma_hybrid_results_dir,
    normalize_split,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import group_by_row, load_jsonl, span_to_bio

BIO_LABELS = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
PAPER_COLOURS = {
    "precision": "#2166ac",
    "recall": "#d6604d",
    "f1": "#4dac26",
    "disease": "#b2182b",
    "chemical": "#2166ac",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def entity_key(entity: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(entity["row_id"]),
        int(entity["start_char"]),
        int(entity["end_char"]),
        str(entity["label"]).upper(),
    )


def build_flat_bio(documents: list[dict], entities: list[dict]) -> list[str]:
    grouped = group_by_row(entities)
    output: list[str] = []
    for document in documents:
        row_id = int(document["row_id"])
        _, labels = span_to_bio(str(document["full_text"]), grouped.get(row_id, []))
        output.extend(labels)
    return output


def find_result(evaluation: dict, name: str) -> dict:
    for row in evaluation.get("results", []):
        if row.get("system") == name:
            return row
    raise KeyError(f"Missing system in MedGemma evaluation: {name}")


def plot_confusion(true_labels: list[str], predicted_labels: list[str], output_file: Path, title: str) -> dict:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=BIO_LABELS)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(BIO_LABELS)))
    axis.set_yticks(np.arange(len(BIO_LABELS)))
    axis.set_xticklabels(BIO_LABELS, rotation=35, ha="right")
    axis.set_yticklabels(BIO_LABELS)
    axis.set_xlabel("Predicted BIO label")
    axis.set_ylabel("Human-gold BIO label")
    axis.set_title(title + "\nRow-normalised; raw token counts shown")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = int(matrix[row, column])
            pct = float(normalized[row, column]) * 100
            axis.text(column, row, f"{count:,}\n{pct:.1f}%", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion within true label", rotation=270, labelpad=18)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"labels": BIO_LABELS, "raw_counts": matrix.tolist(), "row_normalized": normalized.tolist()}


def plot_per_label(evaluation: dict, output_file: Path, split: str) -> None:
    names = ["Weighted Pseudo-Gold", "scispaCy", "MedGemma Zero-Shot", "Weighted + MedGemma Tie-Breaker"]
    rows = [find_result(evaluation, name) for name in names]
    disease = [float(row["per_label"]["DISEASE"]["f1-score"]) for row in rows]
    chemical = [float(row["per_label"]["CHEMICAL"]["f1-score"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10.5, 6))
    d = axis.bar(x - width / 2, disease, width, label="DISEASE", color=PAPER_COLOURS["disease"])
    c = axis.bar(x + width / 2, chemical, width, label="CHEMICAL", color=PAPER_COLOURS["chemical"])
    for bars in (d, c):
        for bar in bars:
            axis.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0,3), textcoords="offset points", ha="center", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=18, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Exact-span F1")
    axis.set_title(f"BioRED {split}: F1 by Entity Label")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap(bootstrap: dict, output_file: Path) -> None:
    rows = bootstrap.get("confidence_intervals", [])
    if not rows:
        return
    names = [str(row["system"]) for row in rows]
    estimates = np.array([float(row["f1"]) for row in rows])
    lows = np.array([float(row["ci_low"]) for row in rows])
    highs = np.array([float(row["ci_high"]) for row in rows])
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.errorbar(
        estimates,
        positions,
        xerr=np.array([estimates - lows, highs - estimates]),
        fmt="o",
        capsize=5,
        color=PAPER_COLOURS["precision"],
        ecolor=PAPER_COLOURS["precision"],
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Exact-span F1 with 95% bootstrap interval")
    axis.set_title("BioRED held-out test bootstrap confidence intervals")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_routing(evaluation: dict, output_file: Path, split: str) -> None:
    routing = evaluation.get("hybrid_routing")
    if not isinstance(routing, dict):
        return
    labels = ["Automatic reject", "MedGemma routed", "Automatic accept"]
    values = [
        int(routing["auto_reject_count"]),
        int(routing["routed_count"]),
        int(routing["auto_accept_count"]),
    ]
    total = int(routing["total_unique_candidates"])
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    bars = axis.bar(labels, values, color=[PAPER_COLOURS["recall"], PAPER_COLOURS["precision"], PAPER_COLOURS["f1"]])
    for bar, value in zip(bars, values):
        pct = 100 * value / total if total else 0
        axis.annotate(f"{value:,}\n({pct:.2f}%)", (bar.get_x()+bar.get_width()/2, value), xytext=(0,4), textcoords="offset points", ha="center", fontsize=9)
    axis.set_ylabel("Unique candidate spans")
    axis.set_title(f"BioRED {split}: Selective MedGemma Routing")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def candidate_audit(decisions: list[dict], gold: list[dict], output_csv: Path) -> dict:
    gold_keys = {entity_key(entity) for entity in gold}
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    rows: list[dict] = []
    for decision in decisions:
        key = entity_key(decision)
        is_gold = key in gold_keys
        accepted = str(decision.get("decision", "")).upper() == "ACCEPT"
        if accepted and is_gold:
            outcome = "TP"
        elif accepted and not is_gold:
            outcome = "FP"
        elif not accepted and not is_gold:
            outcome = "TN"
        else:
            outcome = "FN"
        counts[outcome.lower()] += 1
        rows.append(
            {
                "row_id": int(decision["row_id"]),
                "start_char": int(decision["start_char"]),
                "end_char": int(decision["end_char"]),
                "text": str(decision["text"]),
                "label": str(decision["label"]),
                "weighted_score": float(decision["weighted_score"]),
                "decision": str(decision.get("decision", "")),
                "is_exact_human_gold": is_gold,
                "outcome": outcome,
                "reason": str(decision.get("reason", "")),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["row_id"])
        writer.writeheader()
        writer.writerows(rows)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    return {"counts": counts, "accept_precision": precision, "accept_recall": recall, "reject_specificity": specificity}


def process_split(split: str) -> Path:
    split = normalize_split(split)
    output_dir = results_dir(split) / "biored_report"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    evaluation = read_json(require_file(results_dir(split) / "medgemma_evaluation.json", "BioRED MedGemma evaluation"))
    bootstrap_path = results_dir(split) / "medgemma_bootstrap_results.json"
    bootstrap = read_json(bootstrap_path) if bootstrap_path.exists() else None
    docs = load_jsonl(require_file(docs_file(split), "BioRED documents"))
    gold = load_jsonl(require_file(gold_entities_file(split), "BioRED gold"))
    weighted = load_jsonl(require_file(pseudo_gold_file("weighted", split), "weighted pseudo-gold"))
    scispacy = load_jsonl(require_file(prediction_file("scispacy", split), "scispaCy predictions"))
    medgemma = load_jsonl(require_file(prediction_file("medgemma", split), "MedGemma predictions"))
    hybrid = load_jsonl(require_file(medgemma_hybrid_file(split), "hybrid predictions"))

    true_bio = build_flat_bio(docs, gold)
    matrices = {
        "weighted": plot_confusion(true_bio, build_flat_bio(docs, weighted), figure_dir / "cm_biored_weighted_bio.png", f"BioRED {split}: weighted pseudo-gold vs human gold"),
        "scispacy": plot_confusion(true_bio, build_flat_bio(docs, scispacy), figure_dir / "cm_biored_scispacy_bio.png", f"BioRED {split}: scispaCy vs human gold"),
        "medgemma": plot_confusion(true_bio, build_flat_bio(docs, medgemma), figure_dir / "cm_biored_medgemma_zero_shot_bio.png", f"BioRED {split}: MedGemma zero-shot vs human gold"),
        "hybrid": plot_confusion(true_bio, build_flat_bio(docs, hybrid), figure_dir / "cm_biored_medgemma_hybrid_bio.png", f"BioRED {split}: weighted + MedGemma vs human gold"),
    }
    plot_per_label(evaluation, figure_dir / "biored_medgemma_per_label_f1.png", split)
    plot_routing(evaluation, figure_dir / "biored_medgemma_routing.png", split)
    if bootstrap is not None:
        plot_bootstrap(bootstrap, figure_dir / "biored_medgemma_bootstrap_ci.png")

    decision_path = medgemma_hybrid_results_dir(split) / "candidate_decisions.jsonl"
    audit = None
    if decision_path.exists():
        audit = candidate_audit(load_jsonl(decision_path), gold, output_dir / "candidate_decision_audit.csv")

    weighted_row = find_result(evaluation, "Weighted Pseudo-Gold")
    medgemma_row = find_result(evaluation, "MedGemma Zero-Shot")
    hybrid_row = find_result(evaluation, "Weighted + MedGemma Tie-Breaker")
    routing = evaluation.get("hybrid_routing") or {}
    zero = evaluation.get("zero_shot_diagnostics") or {}

    lines = [
        f"# BioRED MedGemma Report — {split}",
        "",
        "Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.",
        "",
        "## Main result",
        "",
        f"- Weighted pseudo-gold F1: {float(weighted_row['f1']):.4f}",
        f"- MedGemma zero-shot F1: {float(medgemma_row['f1']):.4f}",
        f"- Weighted + MedGemma F1: {float(hybrid_row['f1']):.4f}",
        f"- Hybrid minus weighted F1: {float(hybrid_row['f1']) - float(weighted_row['f1']):+.6f}",
    ]
    if routing:
        lines += [
            "",
            "## Routing",
            "",
            f"- Total candidates: {int(routing.get('total_unique_candidates', 0)):,}",
            f"- Routed to MedGemma: {int(routing.get('routed_count', 0)):,}",
            f"- Routing rate: {100 * float(routing.get('routing_rate_over_all_candidates', 0)):.2f}%",
            f"- Tie-break request time: {float(routing.get('tie_break_request_seconds', 0))/60:.1f} minutes",
        ]
    if zero:
        lines += [
            f"- Full zero-shot request time: {float(zero.get('summed_request_seconds', 0))/60:.1f} minutes",
        ]
    if audit:
        lines += [
            "",
            "## Routed-candidate exact-gold audit",
            "",
            f"- Correct accepts: {audit['counts']['tp']}",
            f"- False accepts: {audit['counts']['fp']}",
            f"- Correct rejects: {audit['counts']['tn']}",
            f"- False rejects: {audit['counts']['fn']}",
            f"- Acceptance precision: {100*audit['accept_precision']:.2f}%",
            f"- Rejection specificity: {100*audit['reject_specificity']:.2f}%",
        ]
    lines += [
        "",
        "BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.",
        "",
        "No Ollama or MedGemma inference was run by this report script.",
    ]
    (output_dir / "BIORED_MEDGEMMA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json({"split": split, "confusion_matrices": matrices, "candidate_audit": audit}, output_dir / "biored_report_data.json")
    print(f"Report: {output_dir / 'BIORED_MEDGEMMA_REPORT.md'}")
    print(f"Figures: {figure_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BioRED report from saved outputs.")
    parser.add_argument("--split", default="test", choices=["dev", "test", "development", "both"])
    args = parser.parse_args()
    print("BioRED post-processing only. Ollama will not be called.")
    if args.split == "both":
        process_split("dev")
        process_split("test")
    else:
        process_split(args.split)


if __name__ == "__main__":
    main()
