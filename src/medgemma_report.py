"""Post-process completed MedGemma BC5CDR outputs.

This module does not call Ollama and does not rerun MedGemma. It reads saved
predictions, human gold, evaluation summaries, bootstrap results, and hybrid
candidate decisions to create paper-ready figures and a transparent report.

Run from the project root:

    python -m src.medgemma_report --split test

Optional:

    python -m src.medgemma_report --split dev
    python -m src.medgemma_report --split both
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

from src.experiment_config import (
    docs_file,
    gold_entities_file,
    medgemma_hybrid_file,
    normalize_split,
    prediction_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import group_by_row, load_jsonl, span_to_bio

BIO_LABELS = ["O", "B-DISEASE", "I-DISEASE", "B-CHEMICAL", "I-CHEMICAL"]
SYSTEM_ORDER = [
    "Majority Pseudo-Gold",
    "Weighted Pseudo-Gold",
    "scispaCy",
    "MedGemma Zero-Shot",
    "Weighted + MedGemma Tie-Breaker",
]

# Match the colour palette used by the existing paper figures.
PAPER_COLOURS = {
    "precision": "#2166ac",  # blue
    "recall": "#d6604d",  # coral
    "f1": "#4dac26",  # green
    "disease": "#b2182b",  # dark red
    "chemical": "#2166ac",  # blue
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def entity_key(entity: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(entity["row_id"]),
        int(entity["start_char"]),
        int(entity["end_char"]),
        str(entity["label"]).upper(),
    )


def build_flat_bio(
    documents: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> list[str]:
    grouped = group_by_row(entities)
    flat_labels: list[str] = []

    for document in documents:
        row_id = int(document["row_id"])
        text = str(document["full_text"])
        _, labels = span_to_bio(text, grouped.get(row_id, []))
        flat_labels.extend(labels)

    return flat_labels


def find_system_result(evaluation: dict[str, Any], name: str) -> dict[str, Any]:
    rows = evaluation.get("results", [])
    for row in rows:
        if row.get("system") == name:
            return row
    raise KeyError(f"System '{name}' is missing from medgemma_evaluation.json")


def annotate_bars(axis: Any, bars: Any, decimals: int = 3) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_exact_metric_comparison(
    evaluation: dict[str, Any],
    output_file: Path,
    split: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for system_name in SYSTEM_ORDER:
        rows.append(find_system_result(evaluation, system_name))

    x_values = np.arange(len(rows))
    width = 0.25
    precision = [float(row["precision"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]
    f1_scores = [float(row["f1"]) for row in rows]

    figure, axis = plt.subplots(figsize=(12, 6.5))
    precision_bars = axis.bar(
        x_values - width,
        precision,
        width,
        label="Precision",
        color=PAPER_COLOURS["precision"],
        edgecolor="white",
        linewidth=0.5,
    )
    recall_bars = axis.bar(
        x_values,
        recall,
        width,
        label="Recall",
        color=PAPER_COLOURS["recall"],
        edgecolor="white",
        linewidth=0.5,
    )
    f1_bars = axis.bar(
        x_values + width,
        f1_scores,
        width,
        label="F1",
        color=PAPER_COLOURS["f1"],
        edgecolor="white",
        linewidth=0.5,
    )

    annotate_bars(axis, precision_bars)
    annotate_bars(axis, recall_bars)
    annotate_bars(axis, f1_bars)

    axis.set_xticks(x_values)
    axis.set_xticklabels([row["system"] for row in rows], rotation=18, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Exact-span score")
    axis.set_title(f"BC5CDR {split}: exact character-span and label performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_per_label_f1(
    evaluation: dict[str, Any],
    output_file: Path,
    split: str,
) -> None:
    selected_names = [
        "Weighted Pseudo-Gold",
        "scispaCy",
        "MedGemma Zero-Shot",
        "Weighted + MedGemma Tie-Breaker",
    ]
    rows: list[dict[str, Any]] = []
    for system_name in selected_names:
        rows.append(find_system_result(evaluation, system_name))

    disease_scores: list[float] = []
    chemical_scores: list[float] = []
    for row in rows:
        per_label = row["per_label"]
        disease_scores.append(float(per_label["DISEASE"]["f1-score"]))
        chemical_scores.append(float(per_label["CHEMICAL"]["f1-score"]))

    x_values = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.5, 6))
    disease_bars = axis.bar(
        x_values - width / 2.0,
        disease_scores,
        width,
        label="DISEASE",
        color=PAPER_COLOURS["disease"],
        edgecolor="white",
        linewidth=0.5,
    )
    chemical_bars = axis.bar(
        x_values + width / 2.0,
        chemical_scores,
        width,
        label="CHEMICAL",
        color=PAPER_COLOURS["chemical"],
        edgecolor="white",
        linewidth=0.5,
    )

    annotate_bars(axis, disease_bars)
    annotate_bars(axis, chemical_bars)

    axis.set_xticks(x_values)
    axis.set_xticklabels(selected_names, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Exact-span F1")
    axis.set_title(f"BC5CDR {split}: F1 by entity label")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_bootstrap_intervals(
    bootstrap: dict[str, Any],
    output_file: Path,
) -> None:
    rows = bootstrap.get("confidence_intervals", [])
    if not rows:
        return

    names: list[str] = []
    estimates: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []

    for row in rows:
        name = str(row["system"])
        estimate = float(row["f1"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        names.append(name)
        estimates.append(estimate)
        lower_errors.append(estimate - low)
        upper_errors.append(high - estimate)

    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.errorbar(
        estimates,
        positions,
        xerr=np.array([lower_errors, upper_errors]),
        fmt="o",
        capsize=5,
        color=PAPER_COLOURS["precision"],
        ecolor=PAPER_COLOURS["precision"],
        markerfacecolor=PAPER_COLOURS["precision"],
        markeredgecolor=PAPER_COLOURS["precision"],
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Exact-span F1 with 95% bootstrap interval")
    axis.set_title("Held-out BC5CDR test bootstrap confidence intervals")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_routing_summary(
    evaluation: dict[str, Any],
    output_file: Path,
    split: str,
) -> None:
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

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    bars = axis.bar(
        labels,
        values,
        color=[
            PAPER_COLOURS["recall"],
            PAPER_COLOURS["precision"],
            PAPER_COLOURS["f1"],
        ],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, value in zip(bars, values):
        percentage = 100.0 * value / total if total else 0.0
        axis.annotate(
            f"{value:,}\n({percentage:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_ylabel("Unique candidate spans")
    axis.set_title(f"BC5CDR {split}: selective MedGemma routing")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_bio_confusion_matrix(
    true_labels: list[str],
    predicted_labels: list[str],
    output_file: Path,
    title: str,
) -> dict[str, Any]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "Token-label length mismatch: "
            f"gold={len(true_labels)}, predictions={len(predicted_labels)}"
        )

    matrix = confusion_matrix(true_labels, predicted_labels, labels=BIO_LABELS)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(np.arange(len(BIO_LABELS)))
    axis.set_yticks(np.arange(len(BIO_LABELS)))
    axis.set_xticklabels(BIO_LABELS, rotation=35, ha="right")
    axis.set_yticklabels(BIO_LABELS)
    axis.set_xlabel("Predicted BIO label")
    axis.set_ylabel("Human-gold BIO label")
    axis.set_title(title + "\nRow-normalised; each cell also shows raw token count")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            count = int(matrix[row_index, column_index])
            percentage = float(normalized[row_index, column_index]) * 100.0
            axis.text(
                column_index,
                row_index,
                f"{count:,}\n{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion within true label", rotation=270, labelpad=18)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return {
        "labels": BIO_LABELS,
        "raw_counts": matrix.tolist(),
        "row_normalized": normalized.tolist(),
        "token_count": len(true_labels),
        "caution": (
            "This is a secondary token-level BIO diagnostic. The frequent O class "
            "can dominate counts; primary evaluation remains exact span and label."
        ),
    }


def analyse_candidate_decisions(
    decisions: list[dict[str, Any]],
    gold_entities: list[dict[str, Any]],
    audit_csv: Path,
) -> dict[str, Any]:
    gold_keys = set()
    for entity in gold_entities:
        gold_keys.add(entity_key(entity))

    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    by_label: dict[str, dict[str, int]] = {}
    by_score: dict[str, dict[str, int]] = {}
    audited_rows: list[dict[str, Any]] = []
    false_accept_examples: list[dict[str, Any]] = []
    false_reject_examples: list[dict[str, Any]] = []

    for decision in decisions:
        key = entity_key(decision)
        is_gold = key in gold_keys
        accepted = str(decision.get("decision", "")).upper() == "ACCEPT"

        if accepted and is_gold:
            outcome = "TP"
            counts["tp"] += 1
        elif accepted and not is_gold:
            outcome = "FP"
            counts["fp"] += 1
        elif not accepted and not is_gold:
            outcome = "TN"
            counts["tn"] += 1
        else:
            outcome = "FN"
            counts["fn"] += 1

        label = str(decision["label"]).upper()
        if label not in by_label:
            by_label[label] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        by_label[label][outcome.lower()] += 1

        score_key = f"{float(decision['weighted_score']):.8f}"
        if score_key not in by_score:
            by_score[score_key] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        by_score[score_key][outcome.lower()] += 1

        audited = {
            "row_id": int(decision["row_id"]),
            "start_char": int(decision["start_char"]),
            "end_char": int(decision["end_char"]),
            "text": str(decision["text"]),
            "label": label,
            "weighted_score": float(decision["weighted_score"]),
            "decision": str(decision.get("decision", "")),
            "is_exact_human_gold": is_gold,
            "outcome": outcome,
            "reason": str(decision.get("reason", "")),
        }
        audited_rows.append(audited)

        if outcome == "FP" and len(false_accept_examples) < 20:
            false_accept_examples.append(audited)
        if outcome == "FN" and len(false_reject_examples) < 20:
            false_reject_examples.append(audited)

    fieldnames = [
        "row_id",
        "start_char",
        "end_char",
        "text",
        "label",
        "weighted_score",
        "decision",
        "is_exact_human_gold",
        "outcome",
        "reason",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audited_rows)

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "definition": (
            "A routed candidate is correct only if row_id, start_char, end_char, "
            "and entity label exactly match BC5CDR human gold."
        ),
        "counts": counts,
        "accuracy": accuracy,
        "accept_precision": precision,
        "accept_recall": recall,
        "accept_f1": f1,
        "reject_specificity": specificity,
        "by_label": by_label,
        "by_weighted_score": by_score,
        "false_accept_examples": false_accept_examples,
        "false_reject_examples": false_reject_examples,
        "audit_csv": str(audit_csv),
    }


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_float(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def comparison_from_bootstrap(
    bootstrap: dict[str, Any] | None,
    system_1: str,
    system_2: str,
) -> dict[str, Any] | None:
    if bootstrap is None:
        return None
    rows = bootstrap.get("comparisons", [])
    for row in rows:
        if row.get("system_1") == system_1 and row.get("system_2") == system_2:
            return row
    return None


def write_markdown_report(
    path: Path,
    split: str,
    evaluation: dict[str, Any],
    bootstrap: dict[str, Any] | None,
    candidate_audit: dict[str, Any],
) -> None:
    weighted = find_system_result(evaluation, "Weighted Pseudo-Gold")
    scispacy = find_system_result(evaluation, "scispaCy")
    medgemma = find_system_result(evaluation, "MedGemma Zero-Shot")
    hybrid = find_system_result(evaluation, "Weighted + MedGemma Tie-Breaker")

    routing = evaluation.get("hybrid_routing", {})
    zero_diagnostics = evaluation.get("zero_shot_diagnostics", {})
    comparison = comparison_from_bootstrap(
        bootstrap,
        "Weighted + MedGemma Tie-Breaker",
        "Weighted Pseudo-Gold",
    )

    delta_precision = float(hybrid["precision"]) - float(weighted["precision"])
    delta_recall = float(hybrid["recall"]) - float(weighted["recall"])
    delta_f1 = float(hybrid["f1"]) - float(weighted["f1"])
    delta_tp = int(hybrid["tp"]) - int(weighted["tp"])
    delta_fp = int(hybrid["fp"]) - int(weighted["fp"])
    delta_fn = int(hybrid["fn"]) - int(weighted["fn"])

    weighted_disease = float(weighted["per_label"]["DISEASE"]["f1-score"])
    hybrid_disease = float(hybrid["per_label"]["DISEASE"]["f1-score"])
    weighted_chemical = float(weighted["per_label"]["CHEMICAL"]["f1-score"])
    hybrid_chemical = float(hybrid["per_label"]["CHEMICAL"]["f1-score"])

    lines: list[str] = []
    lines.append(f"# MedGemma BC5CDR Post-processing Report — {split}")
    lines.append("")
    lines.append(
        "This report was generated from saved outputs. No Ollama or MedGemma inference was rerun."
    )
    lines.append("")
    lines.append("## Main exact-span results")
    lines.append("")
    lines.append("| System | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for system_name in SYSTEM_ORDER:
        row = find_system_result(evaluation, system_name)
        lines.append(
            "| "
            + system_name
            + " | "
            + format_float(float(row["precision"]))
            + " | "
            + format_float(float(row["recall"]))
            + " | "
            + format_float(float(row["f1"]))
            + " | "
            + f"{int(row['tp']):,}"
            + " | "
            + f"{int(row['fp']):,}"
            + " | "
            + f"{int(row['fn']):,}"
            + " |"
        )

    lines.append("")
    lines.append("## Central finding")
    lines.append("")
    lines.append(
        "The selective MedGemma tie-breaker changed F1 from "
        + format_float(float(weighted["f1"]), 6)
        + " to "
        + format_float(float(hybrid["f1"]), 6)
        + ", a difference of "
        + format_float(delta_f1, 6)
        + "."
    )
    lines.append(
        "It added "
        + str(delta_tp)
        + " true positives and "
        + str(delta_fp)
        + " false positives, while changing false negatives by "
        + str(delta_fn)
        + "."
    )
    lines.append(
        "Precision changed by "
        + format_float(delta_precision, 6)
        + " and recall changed by "
        + format_float(delta_recall, 6)
        + "."
    )
    if comparison is not None:
        lines.append(
            "The paired bootstrap comparison gave raw p = "
            + format_float(float(comparison["raw_p"]), 4)
            + " and Holm-adjusted p = "
            + format_float(float(comparison["holm_p"]), 4)
            + ". The difference was not statistically significant."
        )

    lines.append("")
    lines.append("## Per-label effect")
    lines.append("")
    lines.append(
        "- DISEASE F1: "
        + format_float(weighted_disease)
        + " → "
        + format_float(hybrid_disease)
        + " ("
        + format_float(hybrid_disease - weighted_disease, 6)
        + ")."
    )
    lines.append(
        "- CHEMICAL F1: "
        + format_float(weighted_chemical)
        + " → "
        + format_float(hybrid_chemical)
        + " ("
        + format_float(hybrid_chemical - weighted_chemical, 6)
        + ")."
    )

    lines.append("")
    lines.append("## Zero-shot MedGemma")
    lines.append("")
    lines.append(
        "MedGemma zero-shot achieved precision "
        + format_float(float(medgemma["precision"]))
        + ", recall "
        + format_float(float(medgemma["recall"]))
        + ", and F1 "
        + format_float(float(medgemma["f1"]))
        + ". scispaCy F1 was "
        + format_float(float(scispacy["f1"]))
        + "."
    )

    diagnostics = (
        zero_diagnostics.get("diagnostics", {})
        if isinstance(zero_diagnostics, dict)
        else {}
    )
    accepted_outputs = int(diagnostics.get("accepted_entities", 0))
    exact_offsets = int(diagnostics.get("exact_offset_matches", 0))
    repaired_exact = int(diagnostics.get("repaired_exact_occurrences", 0))
    repaired_casefold = int(diagnostics.get("repaired_casefold_occurrences", 0))
    rejected_unmapped = int(diagnostics.get("rejected_unmapped_text", 0))
    rejected_ambiguous = int(diagnostics.get("rejected_ambiguous_occurrence", 0))
    if accepted_outputs:
        lines.append(
            "Of "
            + f"{accepted_outputs:,}"
            + " accepted extraction outputs before final deduplication, only "
            + f"{exact_offsets:,}"
            + " arrived with already correct offsets. The pipeline repaired "
            + f"{repaired_exact + repaired_casefold:,}"
            + " offsets by matching returned text back to the source."
        )
        lines.append(
            "Unmapped outputs: "
            + f"{rejected_unmapped:,}"
            + "; ambiguous repeated occurrences: "
            + f"{rejected_ambiguous:,}"
            + "."
        )

    lines.append("")
    lines.append("## Hybrid routing and cost")
    lines.append("")
    if isinstance(routing, dict) and routing:
        routed_count = int(routing["routed_count"])
        total_candidates = int(routing["total_unique_candidates"])
        accepted_count = int(routing["medgemma_accept_count"])
        rejected_count = int(routing["medgemma_reject_count"])
        lines.append(
            "The pipeline routed "
            + f"{routed_count:,}"
            + " of "
            + f"{total_candidates:,}"
            + " candidates to MedGemma ("
            + percentage(float(routing["routing_rate_over_all_candidates"]))
            + ")."
        )
        lines.append(
            "MedGemma accepted "
            + f"{accepted_count:,}"
            + " routed candidates and rejected "
            + f"{rejected_count:,}"
            + "."
        )
        zero_seconds = (
            float(zero_diagnostics.get("summed_request_seconds", 0.0))
            if isinstance(zero_diagnostics, dict)
            else 0.0
        )
        tie_seconds = float(routing.get("tie_break_request_seconds", 0.0))
        if zero_seconds > 0.0 and tie_seconds > 0.0:
            reduction = 1.0 - tie_seconds / zero_seconds
            lines.append(
                "Selective tie-breaking used "
                + format_float(tie_seconds / 60.0, 1)
                + " minutes of LLM request time, compared with "
                + format_float(zero_seconds / 60.0, 1)
                + " minutes for full zero-shot extraction on the same split. "
                + "This is approximately "
                + percentage(reduction)
                + " less LLM request time."
            )

    lines.append("")
    lines.append("## Routed-candidate audit against exact human gold")
    lines.append("")
    audit_counts = candidate_audit["counts"]
    lines.append("- Correct accepts (TP): " + f"{int(audit_counts['tp']):,}" + ".")
    lines.append("- False accepts (FP): " + f"{int(audit_counts['fp']):,}" + ".")
    lines.append("- Correct rejects (TN): " + f"{int(audit_counts['tn']):,}" + ".")
    lines.append("- False rejects (FN): " + f"{int(audit_counts['fn']):,}" + ".")
    lines.append(
        "- Acceptance precision: "
        + percentage(float(candidate_audit["accept_precision"]))
        + "."
    )
    lines.append(
        "- Acceptance recall: "
        + percentage(float(candidate_audit["accept_recall"]))
        + "."
    )
    lines.append(
        "- Rejection specificity: "
        + percentage(float(candidate_audit["reject_specificity"]))
        + "."
    )

    lines.append("")
    lines.append("## Important adverse observations")
    lines.append("")
    lines.append(
        "1. The hybrid did not improve held-out exact-span F1. Its tiny negative difference from the weighted baseline was not significant."
    )
    lines.append(
        "2. The tie-breaker was permissive: it accepted many correct candidates, but it also accepted false candidates and rejected relatively few false ones."
    )
    lines.append(
        "3. Disease-label decisions were less reliable than chemical-label decisions. Medically plausible symptoms or conditions can still be false positives relative to the BC5CDR annotation policy."
    )
    lines.append(
        "4. MedGemma did not reliably supply exact character offsets. Text-to-source offset repair was essential and must be reported in the methodology."
    )
    lines.append(
        "5. Zero-shot extraction used 450-character segments to prevent repetitive generation and truncated JSON. This segmentation can reduce broader context and should be listed as a limitation."
    )

    false_accepts = candidate_audit.get("false_accept_examples", [])
    short_fragments: list[str] = []
    for example in false_accepts:
        text = str(example.get("text", ""))
        if len(text.strip()) <= 3 and text not in short_fragments:
            short_fragments.append(text)
        if len(short_fragments) >= 8:
            break
    if short_fragments:
        lines.append(
            "6. Examples of false accepted fragments included: "
            + ", ".join(f"`{text}`" for text in short_fragments)
            + "."
        )

    lines.append("")
    lines.append("## Safe paper interpretation")
    lines.append("")
    lines.append(
        "The result supports the cost-saving part of the architecture but not an accuracy-improvement claim. Selective routing processed only a small fraction of candidates and reduced LLM runtime, yet the current MedGemma adjudicator did not significantly outperform the encoder-only weighted ensemble. The negative result should be reported directly rather than described as an improvement."
    )

    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `figures/medgemma_exact_span_comparison.png`")
    lines.append("- `figures/medgemma_per_label_f1.png`")
    lines.append("- `figures/medgemma_routing.png`")
    lines.append("- `figures/cm_medgemma_zero_shot_bio.png`")
    lines.append("- `figures/cm_medgemma_hybrid_bio.png`")
    if bootstrap is not None:
        lines.append("- `figures/medgemma_bootstrap_ci.png`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_split(split: str) -> Path:
    split = normalize_split(split, allow_train=False)
    output_dir = results_dir(split) / "medgemma_report"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = require_file(
        results_dir(split) / "medgemma_evaluation.json",
        "MedGemma evaluation",
    )
    evaluation = read_json(evaluation_path)

    bootstrap_path = results_dir(split) / "medgemma_bootstrap_results.json"
    bootstrap: dict[str, Any] | None = None
    if bootstrap_path.exists():
        bootstrap = read_json(bootstrap_path)

    documents = load_jsonl(require_file(docs_file(split), "parsed BC5CDR documents"))
    human_gold = load_jsonl(
        require_file(gold_entities_file(split), "BC5CDR human gold")
    )
    medgemma_predictions = load_jsonl(
        require_file(
            prediction_file("medgemma", split), "MedGemma zero-shot predictions"
        )
    )
    hybrid_predictions = load_jsonl(
        require_file(medgemma_hybrid_file(split), "MedGemma hybrid predictions")
    )

    decision_path = require_file(
        results_dir(split) / "medgemma_hybrid" / "candidate_decisions.jsonl",
        "MedGemma candidate decisions",
    )
    decisions = load_jsonl(decision_path)

    true_bio = build_flat_bio(documents, human_gold)
    medgemma_bio = build_flat_bio(documents, medgemma_predictions)
    hybrid_bio = build_flat_bio(documents, hybrid_predictions)

    medgemma_confusion = plot_bio_confusion_matrix(
        true_bio,
        medgemma_bio,
        figure_dir / "cm_medgemma_zero_shot_bio.png",
        f"BC5CDR {split}: MedGemma zero-shot vs human gold",
    )
    hybrid_confusion = plot_bio_confusion_matrix(
        true_bio,
        hybrid_bio,
        figure_dir / "cm_medgemma_hybrid_bio.png",
        f"BC5CDR {split}: weighted + MedGemma tie-breaker vs human gold",
    )

    plot_exact_metric_comparison(
        evaluation,
        figure_dir / "medgemma_exact_span_comparison.png",
        split,
    )
    plot_per_label_f1(
        evaluation,
        figure_dir / "medgemma_per_label_f1.png",
        split,
    )
    plot_routing_summary(
        evaluation,
        figure_dir / "medgemma_routing.png",
        split,
    )
    if bootstrap is not None:
        plot_bootstrap_intervals(
            bootstrap,
            figure_dir / "medgemma_bootstrap_ci.png",
        )

    candidate_audit = analyse_candidate_decisions(
        decisions,
        human_gold,
        output_dir / "candidate_decision_audit.csv",
    )

    report_payload = {
        "split": split,
        "postprocessing_only": True,
        "ollama_called": False,
        "primary_metric": "exact_character_span_and_label",
        "evaluation_source": str(evaluation_path),
        "bootstrap_source": str(bootstrap_path) if bootstrap is not None else None,
        "candidate_decision_audit": candidate_audit,
        "token_level_bio_confusion_matrices": {
            "medgemma_zero_shot": medgemma_confusion,
            "weighted_plus_medgemma": hybrid_confusion,
        },
        "caution": (
            "BIO confusion matrices are secondary diagnostics. Primary results are "
            "exact character-span and label metrics."
        ),
    }
    save_json(report_payload, output_dir / "medgemma_report_data.json")
    write_markdown_report(
        output_dir / "MEDGEMMA_RESULTS_REPORT.md",
        split,
        evaluation,
        bootstrap,
        candidate_audit,
    )

    print("Post-processing completed. No Ollama or MedGemma inference was run.")
    print(f"Report: {output_dir / 'MEDGEMMA_RESULTS_REPORT.md'}")
    print(f"Figures: {figure_dir}")
    print(f"Candidate audit: {output_dir / 'candidate_decision_audit.csv'}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MedGemma BC5CDR report and figures without rerunning the LLM."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "development", "test", "both"],
        help="Use test for the final paper report. 'both' also generates a development report.",
    )
    args = parser.parse_args()

    print(
        "MedGemma post-processing only: saved files will be read; Ollama will not be called."
    )
    if args.split == "both":
        process_split("dev")
        process_split("test")
    else:
        process_split(args.split)


if __name__ == "__main__":
    main()
