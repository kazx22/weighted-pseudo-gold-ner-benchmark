from __future__ import annotations

import argparse
import csv
import os

from src.utils import deduplicate_entities, group_by_row, load_jsonl


def _overlap(a: dict, b: dict) -> bool:
    return (
        str(a["label"]).upper() == str(b["label"]).upper()
        and int(a["start_char"]) < int(b["end_char"])
        and int(a["end_char"]) > int(b["start_char"])
    )


def _maximum_overlap_matches(gold: list[dict], pred: list[dict]) -> int:
    adjacency = []
    for prediction in pred:
        adjacency.append([index for index, target in enumerate(gold) if _overlap(prediction, target)])

    matched_gold: dict[int, int] = {}

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in adjacency[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = pred_index
                return True
        return False

    matched = 0
    for pred_index in range(len(pred)):
        if augment(pred_index, set()):
            matched += 1
    return matched


def relaxed_metrics(gold_entities: list[dict], pred_entities: list[dict]) -> dict:
    gold_grouped = group_by_row(deduplicate_entities(gold_entities))
    pred_grouped = group_by_row(deduplicate_entities(pred_entities))
    row_ids = set(gold_grouped) | set(pred_grouped)

    tp = 0
    total_gold = 0
    total_pred = 0
    for row_id in row_ids:
        gold = gold_grouped.get(row_id, [])
        pred = pred_grouped.get(row_id, [])
        tp += _maximum_overlap_matches(gold, pred)
        total_gold += len(gold)
        total_pred += len(pred)

    fp = total_pred - tp
    fn = total_gold - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaxed same-label overlap evaluation for all five NER systems.")
    parser.add_argument("--dataset", choices=["bc5cdr", "biored"], required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()

    if args.dataset == "bc5cdr":
        from src.experiment_config import MODEL_DISPLAY_NAMES, MODEL_KEYS, gold_entities_file, prediction_file, require_file, results_dir
    else:
        os.environ["BIORED_VARIANT"] = "official"
        from src.biored_config import MODEL_DISPLAY_NAMES, MODEL_KEYS, gold_entities_file, prediction_file, require_file, results_dir

    gold = load_jsonl(require_file(gold_entities_file(args.split), "human gold entities"))
    rows = []
    for key in MODEL_KEYS:
        predicted = load_jsonl(require_file(prediction_file(key, args.split), f"{key} predictions"))
        metrics = relaxed_metrics(gold, predicted)
        rows.append({"model_key": key, "model": MODEL_DISPLAY_NAMES[key], **metrics})
        print(f"{MODEL_DISPLAY_NAMES[key]:24} relaxed overlap F1={metrics['f1']:.4f}")

    out_dir = results_dir(args.split) / "relaxed_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "relaxed_overlap_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
