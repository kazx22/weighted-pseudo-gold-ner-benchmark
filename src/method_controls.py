
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import (
    deduplicate_entities,
    exact_span_metrics,
    group_by_row,
    load_jsonl,
    metrics_from_counts,
    per_document_exact_counts,
    save_jsonl,
)


def _dataset_modules(dataset: str):
    if dataset == "bc5cdr":
        cfg = importlib.import_module("src.experiment_config")
        label = "BC5CDR"
    else:
        cfg = importlib.import_module("src.biored_config")
        label = "BioRED"
    return cfg, label


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


def _materialise(
    votes: dict,
    store: dict,
    *,
    active_models: tuple[str, ...],
    minimum_votes: int | None = None,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
) -> list[dict]:
    if (minimum_votes is None) == (weights is None):
        raise ValueError("Choose exactly one of minimum_votes or weighted thresholding.")
    active = set(active_models)
    output: list[dict] = []
    for key, raw_voters in votes.items():
        voters = sorted(set(raw_voters) & active)
        if not voters:
            continue
        vote_count = len(voters)
        if weights is None:
            score = float(vote_count)
            keep = vote_count >= int(minimum_votes)
        else:
            score = sum(float(weights[m]) for m in voters)
            keep = score + 1e-12 >= float(threshold)
        if not keep:
            continue
        entity = dict(store[key])
        entity["voters"] = voters
        entity["vote_count"] = vote_count
        entity["agreement_score"] = round(vote_count / len(active_models), 6)
        entity["weighted_score"] = round(score, 8)
        output.append(entity)
    output.sort(
        key=lambda e: (
            int(e["row_id"]),
            int(e["start_char"]),
            int(e["end_char"]),
            str(e["label"]),
        )
    )
    return output


def _achievable(weights: dict[str, float], models: tuple[str, ...]) -> list[float]:
    values = [float(weights[m]) for m in models]
    scores: set[float] = set()
    for size in range(1, len(values) + 1):
        for subset in itertools.combinations(values, size):
            scores.add(round(sum(subset), 8))
    return sorted(scores)


def _load_predictions(cfg, split: str, models: tuple[str, ...]) -> dict[str, list[dict]]:
    output = {}
    for model in models:
        path = cfg.prediction_file(model, split)
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        output[model] = load_jsonl(path)
    return output


def _paired_bootstrap(
    docs: list[dict],
    gold: list[dict],
    predictions: dict[str, list[dict]],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    gold_by_row = group_by_row(gold)
    counts = {
        key: per_document_exact_counts(docs, gold_by_row, group_by_row(pred))
        for key, pred in predictions.items()
    }

    def aggregate(items: list[dict], indices: np.ndarray) -> dict:
        tp = fp = fn = 0
        for index in indices:
            row = items[int(index)]
            tp += int(row["tp"])
            fp += int(row["fp"])
            fn += int(row["fn"])
        return metrics_from_counts(tp, fp, fn)

    full = np.arange(len(docs))
    observed = {key: aggregate(items, full) for key, items in counts.items()}
    rng = np.random.default_rng(seed)
    samples = {key: np.zeros(resamples, dtype=float) for key in counts}
    for i in range(resamples):
        indices = rng.integers(0, len(docs), size=len(docs))
        for key, items in counts.items():
            samples[key][i] = float(aggregate(items, indices)["f1"])
    return observed, samples


def _p_value(first: np.ndarray, second: np.ndarray) -> float:
    n = len(first)
    diff = first - second
    lo = (np.count_nonzero(diff <= 0) + 1) / (n + 1)
    hi = (np.count_nonzero(diff >= 0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, original in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[original])
        running = max(running, candidate)
        adjusted[original] = running
    return adjusted


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "biored"], required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg, dataset_label = _dataset_modules(args.dataset)
    models = tuple(cfg.MODEL_KEYS)
    display = cfg.MODEL_DISPLAY_NAMES
    config_path = cfg.frozen_config_file()
    if not config_path.exists():
        raise FileNotFoundError(f"Main weighted development config missing: {config_path}")
    main_config = cfg.load_json(config_path)
    weights = {m: float(main_config["weights"][m]) for m in models}

    dev_gold = load_jsonl(cfg.gold_entities_file("dev"))
    dev_predictions = _load_predictions(cfg, "dev", models)
    dev_votes, dev_store = _vote_table(dev_predictions)

                                                                      
    tuned_rows = []
    for k in range(1, len(models) + 1):
        pseudo = _materialise(
            dev_votes, dev_store, active_models=models, minimum_votes=k
        )
        metrics = exact_span_metrics(dev_gold, pseudo)
        tuned_rows.append({"k": k, **metrics, "entity_count": len(pseudo)})
    tuned_best = max(
        tuned_rows,
        key=lambda row: (float(row["f1"]), float(row["precision"]), int(row["k"])),
    )
    selected_k = int(tuned_best["k"])

                                                                                 
    lomo_dev: list[dict] = []
    lomo_frozen: dict[str, dict] = {}
    for omitted in models:
        active = tuple(model for model in models if model != omitted)
        subset_predictions = {m: dev_predictions[m] for m in active}
        votes, store = _vote_table(subset_predictions)
        threshold_rows = []
        for threshold in _achievable(weights, active):
            pseudo = _materialise(
                votes,
                store,
                active_models=active,
                weights=weights,
                threshold=threshold,
            )
            metrics = exact_span_metrics(dev_gold, pseudo)
            threshold_rows.append(
                {"threshold": threshold, **metrics, "entity_count": len(pseudo)}
            )
        best = max(
            threshold_rows,
            key=lambda row: (
                float(row["f1"]),
                float(row["precision"]),
                float(row["threshold"]),
            ),
        )
        lomo_frozen[omitted] = {
            "active_models": list(active),
            "selected_threshold": float(best["threshold"]),
            "development_metrics": best,
        }
        lomo_dev.append(
            {
                "omitted_model": display[omitted],
                "selected_threshold": float(best["threshold"]),
                "dev_precision": float(best["precision"]),
                "dev_recall": float(best["recall"]),
                "dev_f1": float(best["f1"]),
            }
        )

                                                                              
    test_predictions = _load_predictions(cfg, "test", models)
    test_votes, test_store = _vote_table(test_predictions)
    output_dir = cfg.results_dir("test") / "method_controls"
    gold_output_dir = cfg.pseudo_gold_file("weighted", "test").parent / "method_controls"
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_output_dir.mkdir(parents=True, exist_ok=True)

    tuned_test = _materialise(
        test_votes, test_store, active_models=models, minimum_votes=selected_k
    )
    tuned_path = gold_output_dir / "tuned_unweighted_test_entities.jsonl"
    save_jsonl(tuned_test, tuned_path)

    lomo_test_predictions: dict[str, list[dict]] = {}
    for omitted, frozen in lomo_frozen.items():
        active = tuple(frozen["active_models"])
        subset = {m: test_predictions[m] for m in active}
        votes, store = _vote_table(subset)
        pseudo = _materialise(
            votes,
            store,
            active_models=active,
            weights=weights,
            threshold=float(frozen["selected_threshold"]),
        )
        lomo_test_predictions[omitted] = pseudo
        save_jsonl(pseudo, gold_output_dir / f"lomo_without_{omitted}_test_entities.jsonl")

                                                                                  
    test_gold = load_jsonl(cfg.gold_entities_file("test"))
    weighted_test = load_jsonl(cfg.pseudo_gold_file("weighted", "test"))
    fixed_majority_test = load_jsonl(cfg.pseudo_gold_file("majority", "test"))

    fixed_metrics = exact_span_metrics(test_gold, fixed_majority_test)
    weighted_metrics = exact_span_metrics(test_gold, weighted_test)
    tuned_metrics = exact_span_metrics(test_gold, tuned_test)

    lomo_test_rows = []
    for omitted in models:
        metrics = exact_span_metrics(test_gold, lomo_test_predictions[omitted])
        lomo_test_rows.append(
            {
                "omitted_model": display[omitted],
                "selected_dev_threshold": lomo_frozen[omitted]["selected_threshold"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "delta_f1_vs_full_weighted": float(metrics["f1"] - weighted_metrics["f1"]),
            }
        )

    bootstrap_predictions = {
        "weighted": weighted_test,
        "tuned_unweighted": tuned_test,
        **{f"without_{m}": p for m, p in lomo_test_predictions.items()},
    }
    docs = load_jsonl(cfg.docs_file("test"))
    observed, samples = _paired_bootstrap(
        docs,
        test_gold,
        bootstrap_predictions,
        resamples=args.resamples,
        seed=args.seed,
    )
    tuned_p = _p_value(samples["weighted"], samples["tuned_unweighted"])

    lomo_raw = [
        _p_value(samples["weighted"], samples[f"without_{m}"]) for m in models
    ]
    lomo_holm = _holm(lomo_raw)
    for row, raw, adjusted in zip(lomo_test_rows, lomo_raw, lomo_holm):
        row["raw_p_vs_full_weighted"] = raw
        row["holm_p_vs_full_weighted"] = adjusted
        row["significant_holm_0.05"] = adjusted < 0.05

    payload = {
        "dataset": dataset_label,
        "biored_variant": getattr(cfg, "BIORED_VARIANT", None),
        "primary_metric": "exact_character_span_and_label_micro_f1",
        "fit_split": "dev",
        "applied_split": "test",
        "test_gold_not_read_during_construction": True,
        "tuned_unweighted": {
            "development_candidates": tuned_rows,
            "selected_k": selected_k,
            "selected_dev_metrics": tuned_best,
            "test_metrics": tuned_metrics,
            "weighted_test_metrics": weighted_metrics,
            "fixed_three_of_five_test_metrics": fixed_metrics,
            "weighted_minus_tuned_unweighted_f1": float(
                weighted_metrics["f1"] - tuned_metrics["f1"]
            ),
            "paired_bootstrap_raw_p": tuned_p,
            "resamples": args.resamples,
            "seed": args.seed,
        },
        "leave_one_model_out": {
            "frozen_dev_settings": lomo_frozen,
            "development_summary": lomo_dev,
            "test_summary": lomo_test_rows,
            "holm_family_size": len(models),
            "resamples": args.resamples,
            "seed": args.seed,
        },
    }
    cfg.save_json(payload, output_dir / "method_controls.json")
    _write_csv(output_dir / "tuned_unweighted_dev_sweep.csv", tuned_rows)
    _write_csv(output_dir / "lomo_development.csv", lomo_dev)
    _write_csv(output_dir / "lomo_test.csv", lomo_test_rows)

    print(f"{dataset_label} method controls complete.")
    print(
        f"  Tuned unweighted selected k={selected_k}; test F1={tuned_metrics['f1']:.4f}; "
        f"weighted F1={weighted_metrics['f1']:.4f}; p={tuned_p:.4f}"
    )
    print(f"  LOMO ablations: {len(lomo_test_rows)}")
    print(f"  Results: {output_dir / 'method_controls.json'}")


if __name__ == "__main__":
    main()
