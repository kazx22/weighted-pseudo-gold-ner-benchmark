"""Shared I/O, span matching, and BIO conversion helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ALLOWED_LABELS = {"DISEASE", "CHEMICAL"}


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def save_jsonl(records: Iterable[dict], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_by_row(entities: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for entity in entities:
        grouped[int(entity["row_id"])].append(entity)
    return grouped


def entity_key(entity: dict, *, include_row: bool = True) -> tuple:
    core = (
        int(entity["start_char"]),
        int(entity["end_char"]),
        str(entity["label"]).upper(),
    )
    if include_row:
        return (int(entity["row_id"]),) + core
    return core


def deduplicate_entities(entities: Iterable[dict]) -> list[dict]:
    seen: set[tuple] = set()
    output: list[dict] = []
    for entity in entities:
        key = entity_key(entity)
        if key in seen:
            continue
        seen.add(key)
        output.append(entity)
    return output


def exact_span_counts(
    gold_entities: Iterable[dict],
    pred_entities: Iterable[dict],
    *,
    label: str | None = None,
) -> dict[str, int]:
    """Count exact character-span-and-label matches."""
    wanted = label.upper() if label else None
    gold_set = {
        entity_key(entity)
        for entity in gold_entities
        if wanted is None or str(entity["label"]).upper() == wanted
    }
    pred_set = {
        entity_key(entity)
        for entity in pred_entities
        if wanted is None or str(entity["label"]).upper() == wanted
    }
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return {"tp": tp, "fp": fp, "fn": fn, "support": len(gold_set)}


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "support": int(tp + fn),
    }


def exact_span_metrics(
    gold_entities: Iterable[dict],
    pred_entities: Iterable[dict],
    *,
    label: str | None = None,
) -> dict[str, float | int]:
    counts = exact_span_counts(gold_entities, pred_entities, label=label)
    return metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])


def exact_per_label_report(
    gold_entities: Iterable[dict], pred_entities: Iterable[dict]
) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for label in ("DISEASE", "CHEMICAL"):
        metrics = exact_span_metrics(gold_entities, pred_entities, label=label)
        report[label] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1"],
            "support": metrics["support"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
    return report


def per_document_exact_counts(
    docs: Iterable[dict],
    gold_by_row: dict[int, list[dict]],
    pred_by_row: dict[int, list[dict]],
) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    for doc in docs:
        row_id = int(doc["row_id"])
        counts = exact_span_counts(
            gold_by_row.get(row_id, []), pred_by_row.get(row_id, [])
        )
        output.append(counts)
    return output


def span_to_bio(text: str, entities: Iterable[dict]) -> tuple[list[str], list[str]]:
    """Convert character spans to token-level BIO tags for secondary analyses.

    Primary precision/recall/F1 is calculated with exact character spans. This
    helper is retained for Cohen's kappa and confusion matrices. Token offsets
    come from regex matches, so repeated spaces and newlines remain aligned.
    """
    matches = list(re.finditer(r"\S+", text))
    tokens = [match.group(0) for match in matches]
    labels = ["O"] * len(tokens)

    sorted_entities = sorted(
        deduplicate_entities(entities),
        key=lambda item: (int(item["start_char"]), int(item["end_char"])),
    )

    for entity in sorted_entities:
        label = str(entity["label"]).upper()
        if label not in ALLOWED_LABELS:
            continue
        start = int(entity["start_char"])
        end = int(entity["end_char"])
        first = True

        for index, match in enumerate(matches):
            token_start, token_end = match.span()
            if token_start >= end:
                break
            if token_end <= start:
                continue
            if token_start < end and token_end > start:
                labels[index] = f"{'B' if first else 'I'}-{label}"
                first = False

    return tokens, labels


def build_gold_bio(docs: Iterable[dict], grouped_entities: dict[int, list[dict]]) -> list[dict]:
    output: list[dict] = []
    for doc in docs:
        row_id = int(doc["row_id"])
        tokens, bio_labels = span_to_bio(
            str(doc["full_text"]), grouped_entities.get(row_id, [])
        )
        output.append({"row_id": row_id, "tokens": tokens, "bio_labels": bio_labels})
    return output


def derive_weights(f1: dict[str, float]) -> dict[str, float]:
    total = sum(f1.values())
    if total <= 0:
        raise ValueError("Cannot normalise model weights because their sum is zero.")
    return {model: round(score / total, 6) for model, score in f1.items()}


def remove_TEST(input_file: str | Path, output_file: str | Path) -> None:
    """Compatibility cleanup for older ClinicalBERT output files."""
    records = [
        record
        for record in load_jsonl(input_file)
        if str(record.get("label", "")).upper() != "TEST"
    ]
    save_jsonl(records, output_file)
