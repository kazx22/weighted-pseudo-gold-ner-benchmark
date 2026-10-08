from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from src.pipeline_common import MODEL_KEYS
from src.utils import load_jsonl


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def config_for(dataset: str):
    if dataset == "bc5cdr":
        from src import experiment_config as cfg
        return cfg
    from src import biored_config as cfg
    return cfg


def sentence_crossing_heuristic(surface: str) -> bool:
    if "\n" in surface:
        return True
    if re.search(r"[!?][\"')\]]*\s+(?=[A-Z0-9])", surface):
        return True
    abbreviations = {"st", "dr", "mr", "mrs", "ms", "vs", "fig", "no", "prof", "inc"}
    for match in re.finditer(r"\.\s+(?=[A-Z0-9])", surface):
        prefix = surface[: match.start()].rstrip()
        word_match = re.search(r"([A-Za-z]+)$", prefix)
        if word_match:
            word = word_match.group(1).lower()
            if len(word) <= 3 or word in abbreviations:
                continue
        return True
    return False


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit(dataset: str, split: str, top: int, giant_chars: int, giant_fraction: float) -> tuple[dict, Path]:
    cfg = config_for(dataset)
    split = cfg.normalize_split(split, allow_train=False) if dataset == "bc5cdr" else cfg.normalize_split(split)

    docs = load_jsonl(cfg.require_file(cfg.docs_file(split), f"{dataset} {split} documents"))
    docs_by_row = {int(doc["row_id"]): str(doc["full_text"]) for doc in docs}
    output_dir = cfg.results_dir(split) / "span_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_longest: list[dict] = []
    all_flagged: list[dict] = []
    all_errors: list[dict] = []
    summaries: dict[str, dict] = {}

    for model in MODEL_KEYS:
        predictions = load_jsonl(
            cfg.require_file(cfg.prediction_file(model, split), f"{dataset} {model} {split} predictions")
        )
        lengths: list[int] = []
        token_lengths: list[int] = []
        giant_char_count = 0
        giant_fraction_count = 0
        sentence_crossing_count = 0
        newline_crossing_count = 0
        text_mismatch_count = 0
        invalid_bounds_count = 0
        missing_document_count = 0
        empty_text_count = 0
        unknown_label_count = 0
        model_rows: list[dict] = []

        for entity_index, entity in enumerate(predictions):
            row_id = int(entity.get("row_id", -1))
            stored_text = str(entity.get("text", ""))
            label = str(entity.get("label", "")).upper()
            start = int(entity.get("start_char", -1))
            end = int(entity.get("end_char", -1))
            text = docs_by_row.get(row_id)

            if text is None:
                missing_document_count += 1
                all_errors.append({
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "entity_index": entity_index,
                    "row_id": row_id,
                    "error": "missing_document",
                    "start_char": start,
                    "end_char": end,
                    "stored_text": stored_text,
                    "source_text": "",
                })
                continue

            valid_bounds = 0 <= start < end <= len(text)
            if not valid_bounds:
                invalid_bounds_count += 1
                all_errors.append({
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "entity_index": entity_index,
                    "row_id": row_id,
                    "error": "invalid_bounds",
                    "start_char": start,
                    "end_char": end,
                    "stored_text": stored_text,
                    "source_text": "",
                })
                continue

            source_text = text[start:end]
            char_length = end - start
            token_length = len(re.findall(r"\S+", source_text))
            doc_fraction = char_length / len(text) if text else 0.0
            text_matches = source_text == stored_text
            crosses_sentence = sentence_crossing_heuristic(source_text)
            crosses_newline = "\n" in source_text
            giant_by_chars = char_length >= giant_chars
            giant_by_fraction = doc_fraction >= giant_fraction

            lengths.append(char_length)
            token_lengths.append(token_length)
            giant_char_count += int(giant_by_chars)
            giant_fraction_count += int(giant_by_fraction)
            sentence_crossing_count += int(crosses_sentence)
            newline_crossing_count += int(crosses_newline)
            empty_text_count += int(not stored_text)
            unknown_label_count += int(label not in {"DISEASE", "CHEMICAL"})

            if not text_matches:
                text_mismatch_count += 1
                all_errors.append({
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "entity_index": entity_index,
                    "row_id": row_id,
                    "error": "text_offset_mismatch",
                    "start_char": start,
                    "end_char": end,
                    "stored_text": stored_text,
                    "source_text": source_text,
                })

            audit_row = {
                "dataset": dataset,
                "split": split,
                "model": model,
                "row_id": row_id,
                "label": label,
                "start_char": start,
                "end_char": end,
                "char_length": char_length,
                "token_length": token_length,
                "document_char_length": len(text),
                "span_fraction_of_document": round(doc_fraction, 8),
                "text_matches_offset": text_matches,
                "crosses_sentence_boundary_heuristic": crosses_sentence,
                "crosses_newline": crosses_newline,
                "giant_by_char_threshold": giant_by_chars,
                "giant_by_document_fraction": giant_by_fraction,
                "stored_text": stored_text,
                "source_text": source_text,
            }
            model_rows.append(audit_row)
            if giant_by_chars or giant_by_fraction or crosses_sentence or crosses_newline:
                all_flagged.append(audit_row)

        model_rows.sort(key=lambda row: (int(row["char_length"]), float(row["span_fraction_of_document"])), reverse=True)
        all_longest.extend(model_rows[:max(1, top)])

        hard_errors = invalid_bounds_count + missing_document_count + text_mismatch_count + empty_text_count + unknown_label_count
        warnings = giant_char_count + giant_fraction_count + sentence_crossing_count
        summaries[model] = {
            "prediction_count": len(predictions),
            "valid_span_count": len(lengths),
            "hard_integrity_error_count": hard_errors,
            "invalid_bounds_count": invalid_bounds_count,
            "missing_document_count": missing_document_count,
            "text_offset_mismatch_count": text_mismatch_count,
            "empty_text_count": empty_text_count,
            "unknown_label_count": unknown_label_count,
            "giant_by_char_threshold_count": giant_char_count,
            "giant_by_document_fraction_count": giant_fraction_count,
            "sentence_crossing_heuristic_count": sentence_crossing_count,
            "newline_crossing_count": newline_crossing_count,
            "char_length": {
                "min": min(lengths) if lengths else 0,
                "median": percentile(lengths, 0.50),
                "p95": percentile(lengths, 0.95),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths) if lengths else 0,
            },
            "token_length": {
                "min": min(token_lengths) if token_lengths else 0,
                "median": percentile(token_lengths, 0.50),
                "p95": percentile(token_lengths, 0.95),
                "p99": percentile(token_lengths, 0.99),
                "max": max(token_lengths) if token_lengths else 0,
            },
            "status": "FAIL" if hard_errors else ("WARN" if warnings else "PASS"),
        }

    total_hard = sum(int(row["hard_integrity_error_count"]) for row in summaries.values())
    total_warnings = sum(
        int(row["giant_by_char_threshold_count"])
        + int(row["giant_by_document_fraction_count"])
        + int(row["sentence_crossing_heuristic_count"])
        for row in summaries.values()
    )
    status = "FAIL" if total_hard else ("WARN" if total_warnings else "PASS")

    report = {
        "dataset": dataset,
        "split": split,
        "status": status,
        "models_rerun": False,
        "uses_saved_predictions_only": True,
        "document_count": len(docs),
        "giant_char_threshold": giant_chars,
        "giant_document_fraction_threshold": giant_fraction,
        "hard_integrity_error_count": total_hard,
        "warning_flag_count": total_warnings,
        "hard_integrity_errors": [
            "invalid bounds",
            "missing source document",
            "text[start:end] != stored entity text",
            "empty stored entity text",
            "label outside DISEASE/CHEMICAL",
        ],
        "warning_flags": [
            "span length at or above giant-char threshold",
            "span covers at least the configured fraction of the document",
            "span contains a sentence-boundary heuristic",
        ],
        "models": summaries,
        "note": "Warning flags require inspection; they are not automatically treated as malformed predictions.",
    }

    report_path = output_dir / "span_integrity_audit.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    longest_fields = [
        "dataset", "split", "model", "row_id", "label", "start_char", "end_char",
        "char_length", "token_length", "document_char_length", "span_fraction_of_document",
        "text_matches_offset", "crosses_sentence_boundary_heuristic", "crosses_newline",
        "giant_by_char_threshold", "giant_by_document_fraction", "stored_text", "source_text",
    ]
    write_csv(all_longest, output_dir / "longest_predictions.csv", longest_fields)
    write_csv(all_flagged, output_dir / "flagged_predictions.csv", longest_fields)

    error_fields = [
        "dataset", "split", "model", "entity_index", "row_id", "error",
        "start_char", "end_char", "stored_text", "source_text",
    ]
    write_csv(all_errors, output_dir / "integrity_errors.csv", error_fields)

    print(f"{dataset} {split} span audit: {status}")
    for model in MODEL_KEYS:
        row = summaries[model]
        print(
            f"  {model:<12} n={row['prediction_count']:<6} "
            f"hard={row['hard_integrity_error_count']:<4} "
            f"max_chars={row['char_length']['max']:<4} "
            f"giant_chars={row['giant_by_char_threshold_count']:<4} "
            f"giant_fraction={row['giant_by_document_fraction_count']:<4} "
            f"sentence_flags={row['sentence_crossing_heuristic_count']:<4} "
            f"status={row['status']}"
        )
    print(f"  report:  {report_path}")
    print(f"  longest: {output_dir / 'longest_predictions.csv'}")
    print(f"  flagged: {output_dir / 'flagged_predictions.csv'}")
    print(f"  errors:  {output_dir / 'integrity_errors.csv'}")
    return report, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved prediction span integrity and giant-span outliers.")
    parser.add_argument("--dataset", required=True, choices=["bc5cdr", "biored"])
    parser.add_argument("--split", required=True, choices=["dev", "development", "test"])
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--giant-chars", type=int, default=200)
    parser.add_argument("--giant-fraction", type=float, default=0.20)
    parser.add_argument("--fail-on-integrity-error", action="store_true")
    args = parser.parse_args()

    report, _ = audit(
        args.dataset,
        args.split,
        max(1, args.top),
        max(1, args.giant_chars),
        max(0.0, min(1.0, args.giant_fraction)),
    )
    if args.fail_on_integrity_error and report["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
