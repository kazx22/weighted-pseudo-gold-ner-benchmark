"""
error_taxonomy.py

Qualitative error analysis for the BC5CDR NER comparative evaluation.

Classifies every prediction error against human gold into four types:
  - FALSE_POSITIVE  : model predicted an entity, nothing overlaps in gold
  - FALSE_NEGATIVE  : gold has an entity, model missed it entirely
  - BOUNDARY_ERROR  : partial span overlap but not a complete match
  - TYPE_CONFUSION  : span overlaps correctly but label is wrong

Runs on two models for contrast:
  - SciSpacy     (best overall, F1 0.8959)
  - PubMedBERT   (worst on disease, F1 0.568)

Both evaluated against HUMAN GOLD, not pseudo-gold.

Usage (from project root):
    python src/error_taxonomy.py

Output files written to data/analysis/error_taxonomy/:
    {model}_error_counts.txt        - counts per type per label
    {model}_false_positives.txt     - sampled FP cases with context
    {model}_false_negatives.txt     - sampled FN cases with context
    {model}_boundary_errors.txt     - sampled boundary cases with context
    {model}_type_confusions.txt     - sampled type confusion cases with context
"""

import argparse
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

from src.experiment_config import (
    docs_file,
    gold_entities_file,
    normalize_split,
    prediction_file,
    require_file,
    results_dir,
)

SAMPLE_SIZE = 40  # max samples dumped per error type
CONTEXT_CHARS = 120  # characters of surrounding text to show each side of entity
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# I/O helpers  (mirrors utils.py so behaviour is identical)
# ---------------------------------------------------------------------------


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_row(entities):
    grouped = defaultdict(list)
    for e in entities:
        grouped[e["row_id"]].append(e)
    return grouped


# ---------------------------------------------------------------------------
# Span matching helpers
# ---------------------------------------------------------------------------


def spans_overlap(a_start, a_end, b_start, b_end) -> bool:
    """True if two character spans overlap at all (touching boundaries don't count)."""
    return a_start < b_end and a_end > b_start


def spans_match_exactly(a_start, a_end, b_start, b_end) -> bool:
    return a_start == b_start and a_end == b_end


def get_context(text: str, start: int, end: int, window: int = CONTEXT_CHARS) -> str:
    """Return a snippet of text centred on the entity span."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    snippet = text[left:right]
    # highlight the entity span within the snippet
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


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------


def classify_errors(doc_text: str, gold_ents: list, pred_ents: list) -> dict:
    """
    Compare gold vs predictions for a single document and return four buckets.

    Matching strategy  (mirrors span_to_bio overlap logic in utils.py):
      1. For every predicted entity, look for gold entities that overlap its span.
         - If none overlap           -> FALSE_POSITIVE
         - If one overlaps exactly and label matches -> TRUE_POSITIVE (not recorded)
         - If one overlaps exactly but label differs -> TYPE_CONFUSION
         - If one overlaps but span is not exact     -> BOUNDARY_ERROR
      2. Any gold entity not matched by any prediction -> FALSE_NEGATIVE

    Each error record stores enough to write a readable sample:
      pred_text, pred_label, pred_start, pred_end,
      gold_text (if applicable), gold_label (if applicable),
      context (surrounding sentence window)
    """
    errors = {
        "FALSE_POSITIVE": [],
        "FALSE_NEGATIVE": [],
        "BOUNDARY_ERROR": [],
        "TYPE_CONFUSION": [],
    }

    gold_matched = set()  # indices into gold_ents that a prediction accounts for

    for pred in pred_ents:
        ps, pe = pred["start_char"], pred["end_char"]

        overlapping = [
            (i, g)
            for i, g in enumerate(gold_ents)
            if spans_overlap(ps, pe, g["start_char"], g["end_char"])
        ]

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

        # Pick the best overlapping gold entity (most overlap by character count)
        best_i, best_g = max(
            overlapping,
            key=lambda ig: min(pe, ig[1]["end_char"]) - max(ps, ig[1]["start_char"]),
        )
        gold_matched.add(best_i)

        gs, ge = best_g["start_char"], best_g["end_char"]
        exact = spans_match_exactly(ps, pe, gs, ge)

        if exact and pred["label"] == best_g["label"]:
            pass  # true positive - skip

        elif not exact:
            errors["BOUNDARY_ERROR"].append(
                {
                    "pred_text": pred["text"],
                    "pred_label": pred["label"],
                    "pred_start": ps,
                    "pred_end": pe,
                    "gold_text": best_g["text"],
                    "gold_label": best_g["label"],
                    "gold_start": gs,
                    "gold_end": ge,
                    "context": get_context(doc_text, min(ps, gs), max(pe, ge)),
                }
            )

        else:
            # exact span, wrong label -> type confusion
            errors["TYPE_CONFUSION"].append(
                {
                    "pred_text": pred["text"],
                    "pred_label": pred["label"],
                    "pred_start": ps,
                    "pred_end": pe,
                    "gold_text": best_g["text"],
                    "gold_label": best_g["label"],
                    "context": get_context(doc_text, ps, pe),
                }
            )

    # Any gold entity not touched by any prediction -> false negative
    for i, g in enumerate(gold_ents):
        if i not in gold_matched:
            gs, ge = g["start_char"], g["end_char"]
            errors["FALSE_NEGATIVE"].append(
                {
                    "gold_text": g["text"],
                    "gold_label": g["label"],
                    "gold_start": gs,
                    "gold_end": ge,
                    "context": get_context(doc_text, gs, ge),
                }
            )

    return errors


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def run_analysis(model_name: str, docs: list, gold_grouped: dict, pred_grouped: dict):
    """Run classification over all documents and aggregate."""

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

        for etype, records in doc_errors.items():
            all_errors[etype].extend(records)

    return all_errors


# ---------------------------------------------------------------------------
# Counting per label
# ---------------------------------------------------------------------------


def count_by_label(errors: dict) -> dict:
    """
    Return a nested dict: {error_type: {label: count}}.
    FP and BOUNDARY use pred_label; FN uses gold_label; TYPE_CONFUSION uses both.
    """
    counts = {etype: defaultdict(int) for etype in errors}

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


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

LABEL_FOR_ERROR = {
    "FALSE_POSITIVE": "pred_label",
    "FALSE_NEGATIVE": "gold_label",
    "BOUNDARY_ERROR": "pred_label",
    "TYPE_CONFUSION": "pred_label",
}


def format_record(etype: str, r: dict, idx: int) -> str:
    lines = [f"--- Sample {idx+1} ---"]

    if etype == "FALSE_POSITIVE":
        lines.append(
            f"  PRED : [{r['pred_label']}]  '{r['pred_text']}'  (chars {r['pred_start']}-{r['pred_end']})"
        )
        lines.append(f"  GOLD : (nothing)")
    elif etype == "FALSE_NEGATIVE":
        lines.append(
            f"  GOLD : [{r['gold_label']}]  '{r['gold_text']}'  (chars {r['gold_start']}-{r['gold_end']})"
        )
        lines.append(f"  PRED : (missed)")
    elif etype == "BOUNDARY_ERROR":
        lines.append(
            f"  PRED : [{r['pred_label']}]  '{r['pred_text']}'  (chars {r['pred_start']}-{r['pred_end']})"
        )
        lines.append(
            f"  GOLD : [{r['gold_label']}]  '{r['gold_text']}'  (chars {r['gold_start']}-{r['gold_end']})"
        )
    elif etype == "TYPE_CONFUSION":
        lines.append(f"  PRED : [{r['pred_label']}]  '{r['pred_text']}'")
        lines.append(f"  GOLD : [{r['gold_label']}]  '{r['gold_text']}'")

    lines.append(f"  CTX  : {r['context']}")
    return "\n".join(lines)


def write_sample_file(output_path: Path, etype: str, records: list):
    random.seed(RANDOM_SEED)
    sample = random.sample(records, min(SAMPLE_SIZE, len(records)))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ERROR TYPE: {etype}\n")
        f.write(f"Total instances: {len(records)}  |  Showing: {len(sample)}\n")
        f.write("=" * 80 + "\n\n")
        for i, rec in enumerate(sample):
            f.write(format_record(etype, rec, i) + "\n\n")

    print(f"  Wrote {len(sample)} samples -> {output_path}")


def write_counts_file(output_path: Path, model_name: str, counts: dict, totals: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ERROR COUNTS - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")

        grand_total = sum(len(v) for v in totals.values())
        f.write(f"Grand total errors: {grand_total}\n\n")

        for etype, label_counts in counts.items():
            total_for_type = len(totals[etype])
            f.write(f"{etype}  (n={total_for_type})\n")
            for label, n in sorted(label_counts.items()):
                pct = 100 * n / total_for_type if total_for_type else 0
                f.write(f"    {label:<30} {n:>6}  ({pct:.1f}%)\n")
            f.write("\n")

    print(f"  Wrote counts -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run BC5CDR span-level error taxonomy."
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
        choices=["scispacy", "biobert", "pubmedbert", "clinicalbert", "bioelectra"],
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    output_dir = results_dir(split) / "error_taxonomy"
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_path = require_file(docs_file(split), "parsed documents")
    gold_path = require_file(gold_entities_file(split), "human gold entities")

    print(f"Loading {split} documents...")
    docs = load_jsonl(docs_path)
    print(f"  {len(docs)} documents loaded")

    print("Loading human gold...")
    gold_entities = load_jsonl(gold_path)
    gold_grouped = group_by_row(gold_entities)
    print(f"  {len(gold_entities)} gold entities across {len(gold_grouped)} documents")

    for model_name in args.models:
        pred_path = require_file(
            prediction_file(model_name, split), f"{model_name} predictions"
        )
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*60}")

        pred_entities = load_jsonl(pred_path)
        pred_grouped = group_by_row(pred_entities)
        print(f"  {len(pred_entities)} predicted entities")

        all_errors = run_analysis(model_name, docs, gold_grouped, pred_grouped)
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

    print(f"\nDone. All output is in {output_dir}")


if __name__ == "__main__":
    main()
