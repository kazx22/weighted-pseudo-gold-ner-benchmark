from __future__ import annotations

import argparse
import csv
import json
import os
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

N_BOOTSTRAP = 1000
SEED = 42
VARIANTS = ("WV", "TYPE_ONLY", "LENGTH_ONLY", "FORM_ONLY", "CLASS_WISE_F1", "CAWV")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _entity_key(entity: dict) -> tuple[int, int, int, str]:
    return (
        int(entity["row_id"]),
        int(entity["start_char"]),
        int(entity["end_char"]),
        str(entity["label"]).upper(),
    )


def _load_dataset(dataset: str):
    if dataset == "bc5cdr":
        from src import content_aware_voting as cav
        from src.candidate_gold import build_vote_table, load_model_predictions
        from src.experiment_config import (
            MODEL_KEYS,
            docs_file,
            gold_entities_file,
            pseudo_gold_file,
            require_file,
            results_dir,
            save_json,
        )
        return {
            "cav": cav,
            "MODEL_KEYS": MODEL_KEYS,
            "build_vote_table": build_vote_table,
            "load_model_predictions": load_model_predictions,
            "docs_file": docs_file,
            "gold_entities_file": gold_entities_file,
            "pseudo_gold_file": pseudo_gold_file,
            "require_file": require_file,
            "results_dir": results_dir,
            "save_json": save_json,
            "dataset_name": "BC5CDR",
        }

    os.environ["BIORED_VARIANT"] = "official"
    from src import biored_content_aware_voting as cav
    from src.biored_candidate_gold import build_vote_table, load_model_predictions
    from src.biored_config import (
        MODEL_KEYS,
        docs_file,
        gold_entities_file,
        pseudo_gold_file,
        require_file,
        results_dir,
        save_json,
    )
    return {
        "cav": cav,
        "MODEL_KEYS": MODEL_KEYS,
        "build_vote_table": build_vote_table,
        "load_model_predictions": load_model_predictions,
        "docs_file": docs_file,
        "gold_entities_file": gold_entities_file,
        "pseudo_gold_file": pseudo_gold_file,
        "require_file": require_file,
        "results_dir": results_dir,
        "save_json": save_json,
        "dataset_name": "BioRED",
    }


def _candidate_content_weight(model_key: str, entity: dict, reliability: dict, dimension: str) -> float:
    categories = {
        "entity_type": str(entity["label"]).upper(),
        "span_length": None,
        "mention_form": None,
    }
    text = str(entity.get("text", "")).strip()
    categories["span_length"] = "single_token" if len(text.split()) <= 1 else "multi_token"
    characters = [character for character in text if character.isalnum()]
    has_letter = any(character.isalpha() for character in characters)
    has_digit = any(character.isdigit() for character in characters)
    letters = "".join(character for character in text if character.isalpha())
    if (has_letter and has_digit) or (2 <= len(letters) <= 10 and letters.isupper()):
        categories["mention_form"] = "abbreviation_alphanumeric"
    else:
        categories["mention_form"] = "ordinary"
    return float(reliability[model_key][dimension][categories[dimension]]["f1"])


def _score_variant(
    vote_table: dict,
    entity_store: dict,
    *,
    model_keys: tuple[str, ...],
    global_weights: dict[str, float],
    reliability: dict,
    variant: str,
    alpha: float | None,
) -> list[dict]:
    rows = []
    dim_map = {
        "TYPE_ONLY": "entity_type",
        "LENGTH_ONLY": "span_length",
        "FORM_ONLY": "mention_form",
        "CLASS_WISE_F1": "entity_type",
    }
    dimension = dim_map[variant]

    for key, voters_raw in vote_table.items():
        entity = dict(entity_store[key])
        voters = sorted(voters_raw)
        score = 0.0
        for model_key in voters:
            subgroup = _candidate_content_weight(model_key, entity, reliability, dimension)
            if variant == "CLASS_WISE_F1":
                weight = subgroup
            else:
                weight = float(alpha) * float(global_weights[model_key]) + (1.0 - float(alpha)) * subgroup
            score += weight
        rows.append({
            "key": key,
            "entity": entity,
            "voters": voters,
            "vote_count": len(voters),
            "score": round(score, 8),
        })
    rows.sort(key=lambda row: (-float(row["score"]), row["key"]))
    return rows


def _materialise(scored: list[dict], threshold: float, variant: str, alpha: float | None) -> list[dict]:
    output = []
    for row in scored:
        if float(row["score"]) + 1e-12 < float(threshold):
            continue
        entity = dict(row["entity"])
        entity["ablation_variant"] = variant
        entity["ablation_score"] = float(row["score"])
        if alpha is not None:
            entity["ablation_alpha"] = float(alpha)
        output.append(entity)
    output.sort(key=_entity_key)
    return output


def _best(rows: list[dict], include_alpha: bool) -> dict:
    if include_alpha:
        return max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["alpha"]), float(row["threshold"])))
    return max(rows, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["threshold"])))


def _fit_variant(ctx: dict, variant: str, config: dict, gold_dev: list[dict], vote_table: dict, entity_store: dict) -> dict:
    cav = ctx["cav"]
    global_weights = {key: float(config["global_weights"][key]) for key in ctx["MODEL_KEYS"]}
    reliability = config["content_reliability"]
    alpha_grid = tuple(float(value) for value in config["alpha_grid"])

    all_rows = []
    if variant == "CLASS_WISE_F1":
        scored = _score_variant(vote_table, entity_store, model_keys=ctx["MODEL_KEYS"], global_weights=global_weights, reliability=reliability, variant=variant, alpha=None)
        rows = cav.sweep_thresholds(scored, gold_dev, alpha=0.0)
        for row in rows:
            row["variant"] = variant
        all_rows.extend(rows)
        best = _best(all_rows, include_alpha=False)
        return {"variant": variant, "alpha": None, "threshold": float(best["threshold"]), "dev_metrics": best, "sweep": all_rows}

    for alpha in alpha_grid:
        scored = _score_variant(vote_table, entity_store, model_keys=ctx["MODEL_KEYS"], global_weights=global_weights, reliability=reliability, variant=variant, alpha=alpha)
        rows = cav.sweep_thresholds(scored, gold_dev, alpha=alpha)
        for row in rows:
            row["variant"] = variant
        all_rows.extend(rows)
    best = _best(all_rows, include_alpha=True)
    return {"variant": variant, "alpha": float(best["alpha"]), "threshold": float(best["threshold"]), "dev_metrics": best, "sweep": all_rows}


def _aggregate(counts: list[dict], indices: np.ndarray) -> dict:
    tp = fp = fn = 0
    for index in indices:
        row = counts[int(index)]
        tp += int(row["tp"])
        fp += int(row["fp"])
        fn += int(row["fn"])
    return metrics_from_counts(tp, fp, fn)


def _p_value(first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    n = len(difference)
    lo = (np.count_nonzero(difference <= 0) + 1) / (n + 1)
    hi = (np.count_nonzero(difference >= 0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, original in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[original]))
        running = max(running, candidate)
        adjusted[original] = running
    return adjusted


def _bootstrap(docs: list[dict], gold: list[dict], methods: dict[str, list[dict]], resamples: int, seed: int) -> dict:
    gold_by_row = group_by_row(gold)
    counts = {
        method: per_document_exact_counts(docs, gold_by_row, group_by_row(predicted))
        for method, predicted in methods.items()
    }
    n_docs = len(docs)
    full = np.arange(n_docs)
    observed = {method: _aggregate(rows, full) for method, rows in counts.items()}
    rng = np.random.default_rng(seed)
    samples = {method: np.zeros(resamples, dtype=float) for method in methods}
    for iteration in range(resamples):
        indices = rng.integers(0, n_docs, size=n_docs)
        for method, rows in counts.items():
            samples[method][iteration] = float(_aggregate(rows, indices)["f1"])

    cis = {}
    for method in methods:
        low, high = np.percentile(samples[method], [2.5, 97.5])
        cis[method] = {"observed_f1": float(observed[method]["f1"]), "ci_low": float(low), "ci_high": float(high)}

    comparisons = []
    raw = []
    for baseline in ("WV", "TYPE_ONLY", "LENGTH_ONLY", "FORM_ONLY", "CLASS_WISE_F1"):
        difference = samples["CAWV"] - samples[baseline]
        low, high = np.percentile(difference, [2.5, 97.5])
        p = _p_value(samples["CAWV"], samples[baseline])
        raw.append(p)
        comparisons.append({
            "comparison": f"CAWV_vs_{baseline}",
            "winner": "CAWV" if observed["CAWV"]["f1"] >= observed[baseline]["f1"] else baseline,
            "f1_cawv": float(observed["CAWV"]["f1"]),
            "f1_baseline": float(observed[baseline]["f1"]),
            "f1_difference": float(observed["CAWV"]["f1"] - observed[baseline]["f1"]),
            "difference_ci_low": float(low),
            "difference_ci_high": float(high),
            "raw_p": p,
        })
    for row, p_holm in zip(comparisons, _holm(raw)):
        row["holm_p"] = float(p_holm)
        row["significant_holm_0.05"] = bool(p_holm < 0.05)

    return {
        "method": "paired_document_level_bootstrap",
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence_intervals_95": cis,
        "difference_confidence_intervals_95": "percentile CI of paired F1 differences",
        "multiple_testing_correction": "Holm across five CAWV ablation comparisons",
        "comparisons": comparisons,
    }


def run(dataset: str, resamples: int, seed: int) -> None:
    ctx = _load_dataset(dataset)
    cav = ctx["cav"]
    require_file = ctx["require_file"]
    results_dir = ctx["results_dir"]

    config_path = require_file(cav.cawv_config_file(), "frozen CAWV development configuration")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    gold_dev = load_jsonl(require_file(ctx["gold_entities_file"]("dev"), "development human gold"))
    dev_predictions = ctx["load_model_predictions"]("dev")
    dev_vote_table, dev_entity_store = ctx["build_vote_table"](dev_predictions)

    fitted = {}
    sweep_rows = []
    for variant in ("TYPE_ONLY", "LENGTH_ONLY", "FORM_ONLY", "CLASS_WISE_F1"):
        fitted[variant] = _fit_variant(ctx, variant, config, gold_dev, dev_vote_table, dev_entity_store)
        sweep_rows.extend(fitted[variant]["sweep"])

    test_predictions = ctx["load_model_predictions"]("test")
    test_vote_table, test_entity_store = ctx["build_vote_table"](test_predictions)
    global_weights = {key: float(config["global_weights"][key]) for key in ctx["MODEL_KEYS"]}
    reliability = config["content_reliability"]

    out_dir = results_dir("test") / "cawv" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "WV": load_jsonl(require_file(ctx["pseudo_gold_file"]("weighted", "test"), "weighted-voting test output")),
        "CAWV": load_jsonl(require_file(cav.cawv_pseudo_gold_file("test"), "CAWV test output")),
    }

    for variant in ("TYPE_ONLY", "LENGTH_ONLY", "FORM_ONLY", "CLASS_WISE_F1"):
        fit = fitted[variant]
        scored = _score_variant(
            test_vote_table,
            test_entity_store,
            model_keys=ctx["MODEL_KEYS"],
            global_weights=global_weights,
            reliability=reliability,
            variant=variant,
            alpha=fit["alpha"],
        )
        predicted = _materialise(scored, fit["threshold"], variant, fit["alpha"])
        methods[variant] = predicted
        save_jsonl(predicted, out_dir / f"{variant.lower()}_test_entities.jsonl")

    gold_test = load_jsonl(require_file(ctx["gold_entities_file"]("test"), "test human gold"))
    docs = load_jsonl(require_file(ctx["docs_file"]("test"), "test documents"))
    test_metrics = {method: exact_span_metrics(gold_test, predicted) for method, predicted in methods.items()}
    bootstrap = _bootstrap(docs, gold_test, methods, resamples=resamples, seed=seed)

    selection_rows = []
    for variant, fit in fitted.items():
        selection_rows.append({
            "variant": variant,
            "selected_alpha": "" if fit["alpha"] is None else fit["alpha"],
            "selected_threshold": fit["threshold"],
            "dev_precision": fit["dev_metrics"]["precision"],
            "dev_recall": fit["dev_metrics"]["recall"],
            "dev_f1": fit["dev_metrics"]["f1"],
        })
    selection_rows.append({
        "variant": "CAWV",
        "selected_alpha": config["selected_alpha"],
        "selected_threshold": config["selected_threshold"],
        "dev_precision": config["selected_dev_metrics"]["precision"],
        "dev_recall": config["selected_dev_metrics"]["recall"],
        "dev_f1": config["selected_dev_metrics"]["f1"],
    })

    test_rows = []
    for method in VARIANTS:
        metrics = test_metrics[method]
        test_rows.append({
            "method": method,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        })

    _write_csv(out_dir / "dev_ablation_selection.csv", selection_rows)
    _write_csv(out_dir / "dev_ablation_sweeps.csv", sweep_rows)
    _write_csv(out_dir / "test_ablation_metrics.csv", test_rows)
    _write_csv(out_dir / "paired_bootstrap_comparisons.csv", bootstrap["comparisons"])
    ctx["save_json"]({
        "dataset": ctx["dataset_name"],
        "design": {
            "WV": "frozen global DEV-F1 weighted voting",
            "TYPE_ONLY": "global DEV F1 blended with entity-type DEV F1 using DEV-selected alpha",
            "LENGTH_ONLY": "global DEV F1 blended with span-length DEV F1 using DEV-selected alpha",
            "FORM_ONLY": "global DEV F1 blended with mention-form DEV F1 using DEV-selected alpha",
            "CLASS_WISE_F1": "entity-type DEV F1 only; no global-F1 term",
            "CAWV": "global DEV F1 blended with mean of type, length, and form DEV F1",
        },
        "bootstrap": bootstrap,
        "test_metrics": test_metrics,
    }, out_dir / "cawv_ablation_results.json")

    print(f"{ctx['dataset_name']} CAWV ablation complete -> {out_dir}")
    for row in test_rows:
        print(f"  {row['method']:14} F1={float(row['f1']):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CAWV ablation and class-wise baseline.")
    parser.add_argument("--dataset", choices=["bc5cdr", "biored"], required=True)
    parser.add_argument("--resamples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.dataset, args.resamples, args.seed)


if __name__ == "__main__":
    main()
