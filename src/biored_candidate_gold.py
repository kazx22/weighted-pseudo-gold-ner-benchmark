"""Fit BioRED weighted pseudo-gold on development data and apply it to test."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    frozen_config_file,
    gold_entities_file,
    normalize_split,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    save_json,
)
from src.utils import deduplicate_entities, exact_span_metrics, load_jsonl, save_jsonl


def build_vote_table(model_predictions: dict[str, list[dict]]) -> tuple[dict, dict]:
    vote_table: dict[tuple, set[str]] = defaultdict(set)
    entity_store: dict[tuple, dict] = {}
    for model_key, entities in model_predictions.items():
        for entity in deduplicate_entities(entities):
            key = (
                int(entity["row_id"]),
                int(entity["start_char"]),
                int(entity["end_char"]),
                str(entity["label"]).upper(),
            )
            vote_table[key].add(model_key)
            entity_store.setdefault(key, entity)
    return vote_table, entity_store


def materialise_pseudo_gold(
    vote_table: dict,
    entity_store: dict,
    *,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
    minimum_votes: int | None = None,
) -> list[dict]:
    if (weights is None) == (minimum_votes is None):
        raise ValueError("Use either weighted thresholding or minimum_votes, not both.")

    output: list[dict] = []
    for key, voter_set in vote_table.items():
        voters = sorted(voter_set)
        vote_count = len(voters)
        if weights is not None:
            weighted_score = sum(float(weights[model]) for model in voters)
            keep = weighted_score + 1e-12 >= float(threshold)
        else:
            weighted_score = float(vote_count)
            keep = vote_count >= int(minimum_votes)
        if not keep:
            continue
        entity = dict(entity_store[key])
        entity["voters"] = voters
        entity["vote_count"] = vote_count
        entity["agreement_score"] = round(vote_count / len(MODEL_KEYS), 6)
        entity["weighted_score"] = round(weighted_score, 8)
        output.append(entity)

    output.sort(
        key=lambda item: (
            int(item["row_id"]),
            int(item["start_char"]),
            int(item["end_char"]),
            str(item["label"]),
        )
    )
    return output


def achievable_thresholds(weights: dict[str, float]) -> list[float]:
    values = [float(weights[key]) for key in MODEL_KEYS]
    scores: set[float] = set()
    for size in range(1, len(values) + 1):
        for subset in itertools.combinations(values, size):
            scores.add(round(sum(subset), 8))
    return sorted(scores)


def load_model_predictions(split: str) -> dict[str, list[dict]]:
    predictions: dict[str, list[dict]] = {}
    for model_key in MODEL_KEYS:
        path = require_file(
            prediction_file(model_key, split),
            f"{MODEL_DISPLAY_NAMES[model_key]} BioRED {split} predictions",
        )
        predictions[model_key] = load_jsonl(path)
        print(f"  {MODEL_DISPLAY_NAMES[model_key]:15} {len(predictions[model_key]):>7} entities")
    return predictions


def write_threshold_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "entity_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_on_development() -> dict:
    split = "dev"
    model_predictions = load_model_predictions(split)
    gold_entities = load_jsonl(require_file(gold_entities_file(split), "BioRED development gold"))

    weights: dict[str, float] = {}
    model_metrics: dict[str, dict] = {}
    print("\nBioRED development exact-span weights:")
    for model_key in MODEL_KEYS:
        metrics = exact_span_metrics(gold_entities, model_predictions[model_key])
        weights[model_key] = round(float(metrics["f1"]), 8)
        model_metrics[model_key] = metrics
        print(
            f"  {MODEL_DISPLAY_NAMES[model_key]:15} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )

    if max(weights.values()) <= 0:
        raise ValueError("All BioRED development weights are zero.")

    vote_table, entity_store = build_vote_table(model_predictions)
    threshold_rows: list[dict] = []
    for threshold in achievable_thresholds(weights):
        pseudo = materialise_pseudo_gold(
            vote_table,
            entity_store,
            weights=weights,
            threshold=threshold,
        )
        metrics = exact_span_metrics(gold_entities, pseudo)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": round(float(metrics["precision"]), 8),
                "recall": round(float(metrics["recall"]), 8),
                "f1": round(float(metrics["f1"]), 8),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "entity_count": len(pseudo),
            }
        )

    best = max(
        threshold_rows,
        key=lambda row: (row["f1"], row["precision"], row["threshold"]),
    )
    selected_threshold = float(best["threshold"])
    weighted = materialise_pseudo_gold(
        vote_table,
        entity_store,
        weights=weights,
        threshold=selected_threshold,
    )
    majority = materialise_pseudo_gold(vote_table, entity_store, minimum_votes=3)
    save_jsonl(weighted, pseudo_gold_file("weighted", split))
    save_jsonl(majority, pseudo_gold_file("majority", split))

    config = {
        "schema_version": 1,
        "dataset": "BioRED",
        "task_scope": ["DISEASE", "CHEMICAL"],
        "fit_split": "dev",
        "primary_metric": "exact_character_span_and_label_micro_f1",
        "weight_definition": "raw BioRED development exact-span F1",
        "threshold_selection_rule": "highest F1, then highest precision, then highest threshold",
        "weights": weights,
        "selected_threshold": selected_threshold,
        "selected_threshold_metrics": best,
        "development_model_metrics": model_metrics,
        "threshold_results": threshold_rows,
        "majority_rule": "at least 3 of 5 models",
    }
    save_json(config, frozen_config_file())
    write_threshold_csv(threshold_rows, results_dir("dev") / "threshold_selection.csv")
    print(f"\nFrozen BioRED threshold: {selected_threshold:.8f}")
    print(f"Development weighted F1: {best['f1']:.4f}")
    print(f"Config: {frozen_config_file()}")
    return config


def apply_to_test(config_path: Path) -> None:
    require_file(config_path, "frozen BioRED development configuration")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("fit_split") != "dev":
        raise ValueError("BioRED test must use a development-fitted configuration.")

    weights = {key: float(config["weights"][key]) for key in MODEL_KEYS}
    threshold = float(config["selected_threshold"])
    model_predictions = load_model_predictions("test")
    vote_table, entity_store = build_vote_table(model_predictions)
    weighted = materialise_pseudo_gold(
        vote_table,
        entity_store,
        weights=weights,
        threshold=threshold,
    )
    majority = materialise_pseudo_gold(vote_table, entity_store, minimum_votes=3)
    save_jsonl(weighted, pseudo_gold_file("weighted", "test"))
    save_jsonl(majority, pseudo_gold_file("majority", "test"))
    save_json(
        {
            "dataset": "BioRED",
            "task_scope": ["DISEASE", "CHEMICAL"],
            "applied_split": "test",
            "source_config": str(config_path),
            "weights": weights,
            "selected_threshold": threshold,
            "weighted_entity_count": len(weighted),
            "majority_entity_count": len(majority),
            "test_gold_was_not_read": True,
        },
        results_dir("test") / "pseudo_gold_application.json",
    )
    print("\nApplied frozen BioRED development settings to test predictions.")
    print(f"Weighted entities: {len(weighted)}")
    print(f"Majority entities: {len(majority)}")
    print("BioRED test human gold was not accessed during construction.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit/apply BioRED weighted pseudo-gold.")
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    parser.add_argument("--config", type=Path, default=frozen_config_file())
    args = parser.parse_args()
    split = normalize_split(args.split)
    if split == "dev":
        fit_on_development()
    else:
        apply_to_test(args.config)


if __name__ == "__main__":
    main()
