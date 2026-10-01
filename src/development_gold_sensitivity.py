"""Development-gold budget sensitivity for the weighted ensemble.

This module asks how much labelled development data is needed to calibrate the
weighted pseudo-gold method. It samples DEVELOPMENT DOCUMENTS at 25%, 50%, and
75% using ten fixed random seeds, plus a single 100% run. For every sampled
calibration set it:

1. recomputes each conventional model's exact-span F1 weight;
2. selects the weighted threshold on that sampled development subset only;
3. freezes the resulting weights and threshold;
4. constructs a test pseudo-gold set without reading test human annotations;
5. evaluates that frozen construction on the untouched test set.

No NER model is rerun. The module consumes the saved conventional predictions.

Examples:
    python -m src.development_gold_sensitivity --dataset bc5cdr
    set BIORED_VARIANT=official
    python -m src.development_gold_sensitivity --dataset biored
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import deduplicate_entities, exact_span_metrics, load_jsonl

FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DEFAULT_SEEDS = tuple(range(101, 111))


def _config(dataset: str):
    if dataset == "bc5cdr":
        return importlib.import_module("src.experiment_config"), "BC5CDR"
    return importlib.import_module("src.biored_config"), "BioRED"


def _filter_rows(rows: list[dict], allowed_ids: set[int]) -> list[dict]:
    return [row for row in rows if int(row["row_id"]) in allowed_ids]


def _vote_table(predictions: dict[str, list[dict]]) -> tuple[dict, dict]:
    votes: dict[tuple, set[str]] = defaultdict(set)
    store: dict[tuple, dict] = {}
    for model, entities in predictions.items():
        for entity in deduplicate_entities(entities):
            key = (
                int(entity["row_id"]),
                int(entity["start_char"]),
                int(entity["end_char"]),
                str(entity["label"]).upper(),
            )
            votes[key].add(model)
            store.setdefault(key, entity)
    return votes, store


def _materialise(votes: dict, store: dict, weights: dict[str, float], threshold: float) -> list[dict]:
    output: list[dict] = []
    for key, voters in votes.items():
        score = sum(float(weights[model]) for model in voters)
        if score + 1e-12 < float(threshold):
            continue
        entity = dict(store[key])
        entity["voters"] = sorted(voters)
        entity["vote_count"] = len(voters)
        entity["weighted_score"] = round(float(score), 8)
        output.append(entity)
    output.sort(
        key=lambda row: (
            int(row["row_id"]),
            int(row["start_char"]),
            int(row["end_char"]),
            str(row["label"]),
        )
    )
    return output


def _achievable_thresholds(weights: dict[str, float], model_keys: tuple[str, ...]) -> list[float]:
    values = [float(weights[key]) for key in model_keys]
    scores: set[float] = set()
    for size in range(1, len(values) + 1):
        for subset in itertools.combinations(values, size):
            scores.add(round(sum(subset), 8))
    return sorted(scores)


def _fit_subset(
    gold_subset: list[dict],
    predictions_subset: dict[str, list[dict]],
    model_keys: tuple[str, ...],
) -> tuple[dict[str, float], float, dict]:
    weights: dict[str, float] = {}
    for model in model_keys:
        metrics = exact_span_metrics(gold_subset, predictions_subset[model])
        weights[model] = round(float(metrics["f1"]), 8)

    if max(weights.values()) <= 0:
        raise ValueError("All sampled development weights are zero.")

    votes, store = _vote_table(predictions_subset)
    candidates: list[dict] = []
    for threshold in _achievable_thresholds(weights, model_keys):
        pseudo = _materialise(votes, store, weights, threshold)
        metrics = exact_span_metrics(gold_subset, pseudo)
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "entity_count": len(pseudo),
            }
        )

    best = max(
        candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["threshold"]),
        ),
    )
    return weights, float(best["threshold"]), best


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-gold budget sensitivity analysis.")
    parser.add_argument("--dataset", required=True, choices=["bc5cdr", "biored"])
    parser.add_argument("--seeds", type=int, default=10, help="Repeated seeds for 25/50/75%%.")
    args = parser.parse_args()

    cfg, dataset_label = _config(args.dataset)
    model_keys = tuple(cfg.MODEL_KEYS)
    display = dict(cfg.MODEL_DISPLAY_NAMES)

    dev_docs = load_jsonl(cfg.docs_file("dev"))
    dev_gold = load_jsonl(cfg.gold_entities_file("dev"))
    test_predictions = {
        model: load_jsonl(cfg.prediction_file(model, "test")) for model in model_keys
    }
    dev_predictions = {
        model: load_jsonl(cfg.prediction_file(model, "dev")) for model in model_keys
    }

    doc_ids = np.array(sorted(int(row["row_id"]) for row in dev_docs), dtype=np.int64)
    if len(doc_ids) == 0:
        raise ValueError("Development split is empty.")

    test_votes, test_store = _vote_table(test_predictions)
    frozen_test_outputs: list[tuple[dict, list[dict]]] = []
    rows: list[dict] = []
    seeds = DEFAULT_SEEDS[: max(1, min(args.seeds, len(DEFAULT_SEEDS)))]

    # IMPORTANT: test human gold is intentionally not loaded until every
    # frozen test construction below has been completed.
    for fraction in FRACTIONS:
        fraction_seeds = (None,) if fraction == 1.0 else seeds
        for seed in fraction_seeds:
            if fraction == 1.0:
                selected_ids = set(int(value) for value in doc_ids.tolist())
                seed_value = "all"
            else:
                rng = np.random.default_rng(int(seed))
                sample_size = max(1, int(round(len(doc_ids) * fraction)))
                chosen = rng.choice(doc_ids, size=sample_size, replace=False)
                selected_ids = set(int(value) for value in chosen.tolist())
                seed_value = int(seed)

            gold_subset = _filter_rows(dev_gold, selected_ids)
            prediction_subset = {
                model: _filter_rows(dev_predictions[model], selected_ids)
                for model in model_keys
            }
            weights, threshold, dev_metrics = _fit_subset(
                gold_subset,
                prediction_subset,
                model_keys,
            )
            test_pseudo = _materialise(test_votes, test_store, weights, threshold)

            record = {
                "fraction": float(fraction),
                "percent": int(round(fraction * 100)),
                "seed": seed_value,
                "sampled_dev_documents": len(selected_ids),
                "sampled_dev_gold_entities": len(gold_subset),
                "selected_threshold": float(threshold),
                "dev_precision": float(dev_metrics["precision"]),
                "dev_recall": float(dev_metrics["recall"]),
                "dev_f1": float(dev_metrics["f1"]),
                "test_entity_count_constructed_before_gold": len(test_pseudo),
                "weights": weights,
            }
            frozen_test_outputs.append((record, test_pseudo))

    test_gold = load_jsonl(cfg.gold_entities_file("test"))
    flat_rows: list[dict] = []
    detailed_runs: list[dict] = []
    for record, pseudo in frozen_test_outputs:
        test_metrics = exact_span_metrics(test_gold, pseudo)
        full = dict(record)
        full["test_metrics"] = test_metrics
        detailed_runs.append(full)
        flat = {
            key: value for key, value in record.items() if key != "weights"
        }
        for model in model_keys:
            flat[f"weight_{model}"] = float(record["weights"][model])
        flat.update(
            {
                "test_precision": float(test_metrics["precision"]),
                "test_recall": float(test_metrics["recall"]),
                "test_f1": float(test_metrics["f1"]),
                "test_tp": int(test_metrics["tp"]),
                "test_fp": int(test_metrics["fp"]),
                "test_fn": int(test_metrics["fn"]),
            }
        )
        flat_rows.append(flat)

    summary_rows: list[dict] = []
    for fraction in FRACTIONS:
        members = [row for row in flat_rows if float(row["fraction"]) == fraction]
        f1s = np.array([float(row["test_f1"]) for row in members], dtype=float)
        precisions = np.array([float(row["test_precision"]) for row in members], dtype=float)
        recalls = np.array([float(row["test_recall"]) for row in members], dtype=float)
        thresholds = np.array([float(row["selected_threshold"]) for row in members], dtype=float)
        summary_rows.append(
            {
                "fraction": float(fraction),
                "percent": int(round(fraction * 100)),
                "runs": len(members),
                "mean_test_precision": float(precisions.mean()),
                "sd_test_precision": float(precisions.std(ddof=1)) if len(members) > 1 else 0.0,
                "mean_test_recall": float(recalls.mean()),
                "sd_test_recall": float(recalls.std(ddof=1)) if len(members) > 1 else 0.0,
                "mean_test_f1": float(f1s.mean()),
                "sd_test_f1": float(f1s.std(ddof=1)) if len(members) > 1 else 0.0,
                "min_test_f1": float(f1s.min()),
                "max_test_f1": float(f1s.max()),
                "mean_selected_threshold": float(thresholds.mean()),
                "sd_selected_threshold": float(thresholds.std(ddof=1)) if len(members) > 1 else 0.0,
            }
        )

    output_dir = cfg.results_dir("test") / "development_gold_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_config_path = cfg.frozen_config_file()
    main_config = cfg.load_json(main_config_path) if main_config_path.exists() else None
    full_run = next(run for run in detailed_runs if float(run["fraction"]) == 1.0)
    full_matches_main = None
    if main_config:
        threshold_match = abs(
            float(full_run["selected_threshold"]) - float(main_config["selected_threshold"])
        ) <= 1e-8
        weights_match = all(
            abs(float(full_run["weights"][model]) - float(main_config["weights"][model])) <= 1e-8
            for model in model_keys
        )
        full_matches_main = bool(threshold_match and weights_match)

    payload = {
        "dataset": dataset_label,
        "biored_variant": getattr(cfg, "BIORED_VARIANT", None),
        "purpose": "development labelled-data budget sensitivity for weighted pseudo-gold calibration",
        "sampling_unit": "development documents",
        "fractions": list(FRACTIONS),
        "repeated_seeds_for_partial_fractions": list(seeds),
        "full_development_run_repeated_once": True,
        "test_gold_not_read_during_any_pseudo_gold_construction": True,
        "full_development_matches_main_frozen_config": full_matches_main,
        "model_display_names": {key: display[key] for key in model_keys},
        "summary": summary_rows,
        "runs": detailed_runs,
    }
    (output_dir / "development_gold_sensitivity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "development_gold_sensitivity_runs.csv", flat_rows)
    _write_csv(output_dir / "development_gold_sensitivity_summary.csv", summary_rows)

    print(f"{dataset_label} development-gold sensitivity complete.")
    if getattr(cfg, "BIORED_VARIANT", None):
        print(f"  BioRED variant: {cfg.BIORED_VARIANT}")
    for row in summary_rows:
        print(
            f"  {row['percent']:>3}% dev gold: mean test F1={row['mean_test_f1']:.4f} "
            f"(SD={row['sd_test_f1']:.4f}, runs={row['runs']})"
        )
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
