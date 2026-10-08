from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    docs_file,
    gold_entities_file,
    normalize_split,
    prediction_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import deduplicate_entities, exact_span_metrics, group_by_row, load_jsonl, span_to_bio


def token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def entity_token_coverages(text: str, entities: list[dict]) -> list[list[dict]]:
    tokens = token_spans(text)
    coverage: list[list[dict]] = [[] for _ in tokens]
    for entity in deduplicate_entities(entities):
        start = int(entity["start_char"])
        end = int(entity["end_char"])
        label = str(entity["label"]).upper()
        for index, (_, token_start, token_end) in enumerate(tokens):
            if token_start >= end:
                break
            if token_end <= start:
                continue
            if token_start < end and token_end > start:
                coverage[index].append(
                    {
                        "start_char": start,
                        "end_char": end,
                        "label": label,
                        "text": str(entity.get("text", "")),
                    }
                )
    return coverage


def overlap_diagnostics(docs: list[dict], entities_by_row: dict[int, list[dict]]) -> dict:
    token_total = 0
    multi_tokens = 0
    conflicting_type_tokens = 0
    documents_with_multi = 0
    max_cover = 0

    for document in docs:
        row_id = int(document["row_id"])
        coverage = entity_token_coverages(str(document["full_text"]), entities_by_row.get(row_id, []))
        token_total += len(coverage)
        doc_has_multi = False
        for covers in coverage:
            max_cover = max(max_cover, len(covers))
            if len(covers) > 1:
                multi_tokens += 1
                doc_has_multi = True
                if len({item["label"] for item in covers}) > 1:
                    conflicting_type_tokens += 1
        if doc_has_multi:
            documents_with_multi += 1

    return {
        "token_count": token_total,
        "tokens_covered_by_multiple_predicted_entities": multi_tokens,
        "tokens_with_conflicting_entity_types": conflicting_type_tokens,
        "documents_with_overlapping_prediction_tokens": documents_with_multi,
        "maximum_entities_covering_one_token": max_cover,
        "multiple_coverage_token_rate": multi_tokens / token_total if token_total else 0.0,
    }


def flatten_bio(docs: list[dict], entities_by_row: dict[int, list[dict]]) -> tuple[list[str], list[str]]:
    all_tokens: list[str] = []
    all_labels: list[str] = []
    for document in docs:
        row_id = int(document["row_id"])
        tokens, labels = span_to_bio(str(document["full_text"]), entities_by_row.get(row_id, []))
        if len(tokens) != len(labels):
            raise ValueError(f"BIO conversion mismatch for row_id={row_id}")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
    return all_tokens, all_labels


def binary_labels(labels: list[str]) -> list[str]:
    return ["O" if label == "O" else "ENTITY" for label in labels]


def positive_token_metrics(gold: list[str], pred: list[str]) -> dict:
    gold_positive = [label != "O" for label in gold]
    pred_positive = [label != "O" for label in pred]
    tp = sum(g and p for g, p in zip(gold_positive, pred_positive))
    fp = sum((not g) and p for g, p in zip(gold_positive, pred_positive))
    fn = sum(g and (not p) for g, p in zip(gold_positive, pred_positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_model_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_sample_rows(
    docs: list[dict],
    gold_by_row: dict[int, list[dict]],
    pred_by_row: dict[int, list[dict]],
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    for document in docs:
        row_id = int(document["row_id"])
        gold_entities = gold_by_row.get(row_id, [])
        pred_entities = pred_by_row.get(row_id, [])
        if not gold_entities and not pred_entities:
            continue
        gold_tokens, gold_labels = span_to_bio(str(document["full_text"]), gold_entities)
        pred_tokens, pred_labels = span_to_bio(str(document["full_text"]), pred_entities)
        if gold_tokens != pred_tokens:
            raise ValueError(f"Sample token mismatch for row_id={row_id}")
        selected.append(
            {
                "row_id": row_id,
                "tokens": gold_tokens,
                "gold": gold_labels,
                "prediction": pred_labels,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def write_sample_tsv(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_id", "token_index", "token", "gold_bio", "biobert_bio"])
        for sample in samples:
            for index, (token, gold, pred) in enumerate(
                zip(sample["tokens"], sample["gold"], sample["prediction"])
            ):
                writer.writerow([sample["row_id"], index, token, gold, pred])


def main() -> None:
    parser = argparse.ArgumentParser(description="BioRED kappa sanity checks using saved predictions.")
    parser.add_argument("--split", default="test", choices=["dev", "development", "test"])
    parser.add_argument("--sample-docs", type=int, default=5)
    args = parser.parse_args()
    split = normalize_split(args.split)

    docs = load_jsonl(require_file(docs_file(split), "BioRED documents"))
    gold = load_jsonl(require_file(gold_entities_file(split), "BioRED human gold"))
    gold_by_row = group_by_row(gold)
    gold_tokens, gold_bio = flatten_bio(docs, gold_by_row)
    gold_binary = binary_labels(gold_bio)

    rows: list[dict] = []
    detailed: dict[str, dict] = {}
    biobert_predictions: list[dict] | None = None

    for key in MODEL_KEYS:
        predictions = load_jsonl(require_file(prediction_file(key, split), f"{key} predictions"))
        pred_by_row = group_by_row(predictions)
        pred_tokens, pred_bio = flatten_bio(docs, pred_by_row)
        if pred_tokens != gold_tokens:
            raise ValueError(f"Token sequence mismatch for {key}; kappa alignment is invalid.")

        multiclass_kappa = float(cohen_kappa_score(gold_bio, pred_bio))
        pred_binary = binary_labels(pred_bio)
        binary_kappa = float(cohen_kappa_score(gold_binary, pred_binary))
        token_metrics = positive_token_metrics(gold_bio, pred_bio)
        overlaps = overlap_diagnostics(docs, pred_by_row)
        exact = exact_span_metrics(gold, predictions)

        name = MODEL_DISPLAY_NAMES[key]
        detailed[name] = {
            "multiclass_bio_kappa": multiclass_kappa,
            "binary_entity_vs_o_kappa": binary_kappa,
            "positive_token_metrics": token_metrics,
            "overlap_diagnostics": overlaps,
            "exact_span_metrics": exact,
            "bio_label_counts": dict(Counter(pred_bio)),
        }
        rows.append(
            {
                "model": name,
                "exact_f1": round(float(exact["f1"]), 8),
                "exact_tp": int(exact["tp"]),
                "multiclass_bio_kappa": round(multiclass_kappa, 8),
                "binary_entity_vs_o_kappa": round(binary_kappa, 8),
                "positive_token_precision": round(float(token_metrics["precision"]), 8),
                "positive_token_recall": round(float(token_metrics["recall"]), 8),
                "positive_token_f1": round(float(token_metrics["f1"]), 8),
                "multi_covered_tokens": int(overlaps["tokens_covered_by_multiple_predicted_entities"]),
                "conflicting_type_tokens": int(overlaps["tokens_with_conflicting_entity_types"]),
                "docs_with_overlap": int(overlaps["documents_with_overlapping_prediction_tokens"]),
            }
        )
        if key == "biobert":
            biobert_predictions = predictions

    if biobert_predictions is None:
        raise RuntimeError("BioBERT predictions were not loaded.")

    output_dir = results_dir(split) / "kappa_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_model_summary(rows, output_dir / "kappa_model_summary.csv")

    samples = select_sample_rows(
        docs,
        gold_by_row,
        group_by_row(biobert_predictions),
        max(1, int(args.sample_docs)),
    )
    write_sample_tsv(samples, output_dir / "biobert_gold_bio_samples.tsv")

    report = {
        "dataset": "BioRED",
        "split": split,
        "uses_saved_predictions_only": True,
        "models_rerun": False,
        "token_count": len(gold_tokens),
        "sample_document_count": len(samples),
        "checks": {
            "overlapping_prediction_tokens": True,
            "gold_vs_biobert_bio_samples": True,
            "binary_entity_vs_o_kappa": True,
            "positive_token_precision_recall": True,
        },
        "models": detailed,
        "interpretation_note": (
            "This diagnostic does not change model predictions. It checks whether unusual token-level kappa values "
            "can be explained by BIO conversion, overlapping predicted spans, or positive-token behaviour."
        ),
    }
    save_json(report, output_dir / "kappa_diagnostic.json")

    print("BioRED kappa diagnostic complete.\n")
    for row in rows:
        print(
            f"{row['model']:<24} exactF1={row['exact_f1']:.4f}  "
            f"BIO kappa={row['multiclass_bio_kappa']:.4f}  "
            f"binary kappa={row['binary_entity_vs_o_kappa']:.4f}  "
            f"token P/R={row['positive_token_precision']:.4f}/{row['positive_token_recall']:.4f}  "
            f"multi-cover tokens={row['multi_covered_tokens']}"
        )
    print(f"\nSaved: {output_dir / 'kappa_diagnostic.json'}")
    print(f"Saved: {output_dir / 'kappa_model_summary.csv'}")
    print(f"Saved: {output_dir / 'biobert_gold_bio_samples.tsv'}")


if __name__ == "__main__":
    main()
