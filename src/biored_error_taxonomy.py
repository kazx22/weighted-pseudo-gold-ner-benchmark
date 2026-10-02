
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    docs_file,
    gold_entities_file,
    normalize_split,
    prediction_file,
    require_file,
    results_dir,
)

SAMPLE_SIZE = 40
CONTEXT_CHARS = 120
RANDOM_SEED = 42


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_row(entities):
    grouped = defaultdict(list)
    for entity in entities:
        grouped[entity["row_id"]].append(entity)
    return grouped


def spans_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and a_end > b_start


def spans_match_exactly(a_start, a_end, b_start, b_end) -> bool:
    return a_start == b_start and a_end == b_end


def get_context(text: str, start: int, end: int, window: int = CONTEXT_CHARS) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    snippet = text[left:right]

    offset = start - left
    entity_len = end - start
    highlighted = (
        snippet[:offset]
        + ">>>"
        + snippet[offset : offset + entity_len]
        + "<<<"
        + snippet[offset + entity_len :]
    )
    return prefix + highlighted + suffix


def classify_errors(doc_text: str, gold_ents: list, pred_ents: list) -> dict:
    errors = {
        "FALSE_POSITIVE": [],
        "FALSE_NEGATIVE": [],
        "BOUNDARY_ERROR": [],
        "TYPE_CONFUSION": [],
    }

    gold_matched = set()

    for pred in pred_ents:
        ps = pred["start_char"]
        pe = pred["end_char"]

        overlapping = []
        for i, gold in enumerate(gold_ents):
            if spans_overlap(ps, pe, gold["start_char"], gold["end_char"]):
                overlapping.append((i, gold))

        if not overlapping:
            errors["FALSE_POSITIVE"].append(
                {
                    "pred_text": pred["text"],
                    "pred_label": pred["label"],
                    "pred_start": ps,
                    "pred_end": pe,
                    "context": get_context(doc_text, ps, pe),
                }
            )
            continue

        best_i, best_gold = max(
            overlapping,
            key=lambda item: min(pe, item[1]["end_char"])
            - max(ps, item[1]["start_char"]),
        )
        gold_matched.add(best_i)

        gs = best_gold["start_char"]
        ge = best_gold["end_char"]
        exact = spans_match_exactly(ps, pe, gs, ge)

        if exact and pred["label"] == best_gold["label"]:
            continue

        if not exact:
            errors["BOUNDARY_ERROR"].append(
                {
                    "pred_text": pred["text"],
                    "pred_label": pred["label"],
                    "pred_start": ps,
                    "pred_end": pe,
                    "gold_text": best_gold["text"],
                    "gold_label": best_gold["label"],
                    "gold_start": gs,
                    "gold_end": ge,
                    "context": get_context(doc_text, min(ps, gs), max(pe, ge)),
                }
            )
        else:
            errors["TYPE_CONFUSION"].append(
                {
                    "pred_text": pred["text"],
                    "pred_label": pred["label"],
                    "pred_start": ps,
                    "pred_end": pe,
                    "gold_text": best_gold["text"],
                    "gold_label": best_gold["label"],
                    "context": get_context(doc_text, ps, pe),
                }
            )

    for i, gold in enumerate(gold_ents):
        if i not in gold_matched:
            gs = gold["start_char"]
            ge = gold["end_char"]
            errors["FALSE_NEGATIVE"].append(
                {
                    "gold_text": gold["text"],
                    "gold_label": gold["label"],
                    "gold_start": gs,
                    "gold_end": ge,
                    "context": get_context(doc_text, gs, ge),
                }
            )

    return errors


def run_analysis(docs: list, gold_grouped: dict, pred_grouped: dict):
    all_errors = {
        "FALSE_POSITIVE": [],
        "FALSE_NEGATIVE": [],
        "BOUNDARY_ERROR": [],
        "TYPE_CONFUSION": [],
    }

    for doc in docs:
        row_id = doc["row_id"]
        text = doc["full_text"]
        gold = gold_grouped.get(row_id, [])
        pred = pred_grouped.get(row_id, [])

        doc_errors = classify_errors(text, gold, pred)
        for error_type, records in doc_errors.items():
            all_errors[error_type].extend(records)

    return all_errors


def count_by_label(errors: dict) -> dict:
    counts = {error_type: defaultdict(int) for error_type in errors}

    for record in errors["FALSE_POSITIVE"]:
        counts["FALSE_POSITIVE"][record["pred_label"]] += 1

    for record in errors["FALSE_NEGATIVE"]:
        counts["FALSE_NEGATIVE"][record["gold_label"]] += 1

    for record in errors["BOUNDARY_ERROR"]:
        counts["BOUNDARY_ERROR"][record["pred_label"]] += 1

    for record in errors["TYPE_CONFUSION"]:
        key = f"{record['gold_label']}->{record['pred_label']}"
        counts["TYPE_CONFUSION"][key] += 1

    return counts


def format_record(error_type: str, record: dict, idx: int) -> str:
    lines = [f"--- Sample {idx + 1} ---"]

    if error_type == "FALSE_POSITIVE":
        lines.append(
            f"  PRED : [{record['pred_label']}]  '{record['pred_text']}'  "
            f"(chars {record['pred_start']}-{record['pred_end']})"
        )
        lines.append("  GOLD : (nothing)")

    elif error_type == "FALSE_NEGATIVE":
        lines.append(
            f"  GOLD : [{record['gold_label']}]  '{record['gold_text']}'  "
            f"(chars {record['gold_start']}-{record['gold_end']})"
        )
        lines.append("  PRED : (missed)")

    elif error_type == "BOUNDARY_ERROR":
        lines.append(
            f"  PRED : [{record['pred_label']}]  '{record['pred_text']}'  "
            f"(chars {record['pred_start']}-{record['pred_end']})"
        )
        lines.append(
            f"  GOLD : [{record['gold_label']}]  '{record['gold_text']}'  "
            f"(chars {record['gold_start']}-{record['gold_end']})"
        )

    elif error_type == "TYPE_CONFUSION":
        lines.append(
            f"  PRED : [{record['pred_label']}]  '{record['pred_text']}'"
        )
        lines.append(
            f"  GOLD : [{record['gold_label']}]  '{record['gold_text']}'"
        )

    lines.append(f"  CTX  : {record['context']}")
    return "\n".join(lines)


def write_sample_file(output_path: Path, error_type: str, records: list):
    random.seed(RANDOM_SEED)
    sample = random.sample(records, min(SAMPLE_SIZE, len(records)))

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"ERROR TYPE: {error_type}\n")
        handle.write(f"Total instances: {len(records)}  |  Showing: {len(sample)}\n")
        handle.write("=" * 80 + "\n\n")

        for i, record in enumerate(sample):
            handle.write(format_record(error_type, record, i) + "\n\n")

    print(f"  Wrote {len(sample)} samples -> {output_path}")


def write_counts_file(output_path: Path, model_name: str, counts: dict, totals: dict):
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"BIORED ERROR COUNTS - {display_name.upper()}\n")
        handle.write("=" * 60 + "\n\n")

        grand_total = sum(len(records) for records in totals.values())
        handle.write(f"Grand total errors: {grand_total}\n\n")

        for error_type, label_counts in counts.items():
            total_for_type = len(totals[error_type])
            handle.write(f"{error_type}  (n={total_for_type})\n")

            for label, count in sorted(label_counts.items()):
                pct = 100 * count / total_for_type if total_for_type else 0
                handle.write(f"    {label:<30} {count:>6}  ({pct:.1f}%)\n")

            handle.write("\n")

    print(f"  Wrote counts -> {output_path}")


def build_summary(split: str, model_name: str, errors: dict, counts: dict) -> dict:
    totals = {error_type: len(records) for error_type, records in errors.items()}
    grand_total = sum(totals.values())

    by_label = {}
    for error_type, label_counts in counts.items():
        by_label[error_type] = dict(label_counts)

    return {
        "dataset": "BioRED",
        "task_scope": ["DISEASE", "CHEMICAL"],
        "split": split,
        "model_key": model_name,
        "model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "taxonomy": {
            "FALSE_POSITIVE": "prediction has no overlapping gold entity",
            "FALSE_NEGATIVE": "gold entity has no overlapping prediction",
            "BOUNDARY_ERROR": "prediction overlaps gold but span is not exact",
            "TYPE_CONFUSION": "exact span but wrong target label",
        },
        "totals": totals,
        "grand_total": grand_total,
        "percent_of_taxonomy_total": {
            error_type: (100.0 * count / grand_total if grand_total else 0.0)
            for error_type, count in totals.items()
        },
        "by_label": by_label,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run BioRED disease/chemical span-level error taxonomy."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test", "development"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["scispacy", "pubmedbert"],
        choices=["scispacy", "biobert", "pubmedbert", "clinicalbert", "d4data"],
    )
    args = parser.parse_args()
    split = normalize_split(args.split)

    output_dir = results_dir(split) / "error_taxonomy"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = require_file(docs_file(split), "BioRED parsed documents")
    gold_path = require_file(gold_entities_file(split), "BioRED human gold entities")

    print(f"Loading BioRED {split} documents...")
    docs = load_jsonl(docs_path)
    print(f"  {len(docs)} documents loaded")

    print("Loading BioRED human gold...")
    gold_entities = load_jsonl(gold_path)
    gold_grouped = group_by_row(gold_entities)
    print(f"  {len(gold_entities)} gold entities across {len(gold_grouped)} documents")

    summaries = []

    for model_name in args.models:
        pred_path = require_file(
            prediction_file(model_name, split),
            f"BioRED {model_name} predictions",
        )

        print(f"\n{'=' * 60}")
        print(f"MODEL: {MODEL_DISPLAY_NAMES.get(model_name, model_name).upper()}")
        print(f"{'=' * 60}")

        pred_entities = load_jsonl(pred_path)
        pred_grouped = group_by_row(pred_entities)
        print(f"  {len(pred_entities)} predicted entities")

        all_errors = run_analysis(docs, gold_grouped, pred_grouped)

        for error_type, records in all_errors.items():
            print(f"  {error_type:<20} {len(records):>6} instances")

        counts = count_by_label(all_errors)

        write_counts_file(
            output_dir / f"{model_name}_error_counts.txt",
            model_name,
            counts,
            all_errors,
        )

        for error_type, records in all_errors.items():
            if records:
                write_sample_file(
                    output_dir / f"{model_name}_{error_type.lower()}.txt",
                    error_type,
                    records,
                )

        summaries.append(build_summary(split, model_name, all_errors, counts))

    summary_path = output_dir / "biored_error_taxonomy_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("BioRED Error Taxonomy Summary\n")
        handle.write("=" * 60 + "\n")
        handle.write("Dataset: BioRED\n")
        handle.write("Task scope: DISEASE and CHEMICAL\n")
        handle.write(f"Split: {split}\n")
        handle.write("Matching logic: same as src/error_taxonomy.py for BC5CDR\n\n")

        for summary in summaries:
            handle.write(f"Model: {summary['model']}\n")
            handle.write("-" * 60 + "\n")
            handle.write(f"Total taxonomy events: {summary['grand_total']}\n")

            for error_type in [
                "FALSE_POSITIVE",
                "FALSE_NEGATIVE",
                "BOUNDARY_ERROR",
                "TYPE_CONFUSION",
            ]:
                count = summary["totals"].get(error_type, 0)
                pct = summary["percent_of_taxonomy_total"].get(error_type, 0.0)
                handle.write(f"{error_type}: {count} ({pct:.1f}%)\n")

                label_counts = summary["by_label"].get(error_type, {})
                for label, label_count in sorted(label_counts.items()):
                    handle.write(f"    {label}: {label_count}\n")

            handle.write("\n")

    print(f"\nSaved text summary -> {summary_path}")
    print(f"Done. All BioRED taxonomy output is in {output_dir}")


if __name__ == "__main__":
    main()
