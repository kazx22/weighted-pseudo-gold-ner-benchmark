
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from src.biored_candidate_gold import build_vote_table, load_model_predictions
from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    GOLD_DIR,
    frozen_config_file,
    gold_entities_file,
    normalize_split,
    require_file,
    results_dir,
    save_json,
)
from src.utils import (
    deduplicate_entities,
    exact_span_metrics,
    load_jsonl,
    save_jsonl,
)

CWA_SCHEMA_VERSION = 1
# Both global and content-specific reliability stay active.
DEFAULT_ALPHA_GRID = tuple(round(i / 10.0, 1) for i in range(1, 10))


def cwa_results_dir(split: str) -> Path:
    return results_dir(split) / "cwa"


def cwa_config_file() -> Path:
    return cwa_results_dir("dev") / "frozen_cwa_config.json"


def cwa_pseudo_gold_file(split: str) -> Path:
    split = normalize_split(split, allow_train=False)
    return GOLD_DIR / f"cwa_pseudo_gold_{split}_entities_biored.jsonl"


def _entity_key(entity: dict) -> tuple[int, int, int, str]:
    return (
        int(entity["row_id"]),
        int(entity["start_char"]),
        int(entity["end_char"]),
        str(entity["label"]).upper(),
    )


def _surface(entity: dict) -> str:
    text = str(entity.get("text", "")).strip()
    if not text:
        raise ValueError(
            "CWA requires each entity record to contain its surface text in the 'text' field. "
            f"Missing text for {_entity_key(entity)}"
        )
    return text


def span_length_category(entity: dict) -> str:
    token_count = len(re.findall(r"\S+", _surface(entity)))
    return "single_token" if token_count <= 1 else "multi_token"


def mention_form_category(entity: dict) -> str:
    text = _surface(entity)
    characters = [character for character in text if character.isalnum()]
    has_letter = any(character.isalpha() for character in characters)
    has_digit = any(character.isdigit() for character in characters)
    if has_letter and has_digit:
        return "abbreviation_alphanumeric"

    letters = "".join(character for character in text if character.isalpha())
    if 2 <= len(letters) <= 10 and letters.isupper():
        return "abbreviation_alphanumeric"

    return "ordinary"


def content_categories(entity: dict) -> dict[str, str]:
    label = str(entity["label"]).upper()
    if label not in {"DISEASE", "CHEMICAL"}:
        raise ValueError(f"Unexpected BioRED label for CWA: {label}")
    return {
        "entity_type": label,
        "span_length": span_length_category(entity),
        "mention_form": mention_form_category(entity),
    }


def _filter_entities(
    entities: list[dict],
    dimension: str,
    category: str,
) -> list[dict]:
    output: list[dict] = []
    for entity in deduplicate_entities(entities):
        if content_categories(entity)[dimension] == category:
            output.append(entity)
    return output


def learn_content_reliability(
    gold_entities: list[dict],
    model_predictions: dict[str, list[dict]],
    global_weights: dict[str, float],
) -> tuple[dict[str, dict], list[dict]]:
    categories = {
        "entity_type": ("DISEASE", "CHEMICAL"),
        "span_length": ("single_token", "multi_token"),
        "mention_form": ("ordinary", "abbreviation_alphanumeric"),
    }

    reliability: dict[str, dict] = {}
    csv_rows: list[dict] = []

    for model_key in MODEL_KEYS:
        reliability[model_key] = {}
        for dimension, values in categories.items():
            reliability[model_key][dimension] = {}
            for category in values:
                gold_subset = _filter_entities(gold_entities, dimension, category)
                pred_subset = _filter_entities(
                    model_predictions[model_key], dimension, category
                )
                metrics = exact_span_metrics(gold_subset, pred_subset)

                                                                              
                                                                         
                used_fallback = int(metrics["support"]) == 0
                f1_value = (
                    float(global_weights[model_key])
                    if used_fallback
                    else float(metrics["f1"])
                )

                reliability[model_key][dimension][category] = {
                    "f1": round(f1_value, 8),
                    "raw_f1": round(float(metrics["f1"]), 8),
                    "precision": round(float(metrics["precision"]), 8),
                    "recall": round(float(metrics["recall"]), 8),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "support": int(metrics["support"]),
                    "fallback_to_global_f1": used_fallback,
                }
                csv_rows.append(
                    {
                        "model_key": model_key,
                        "model": MODEL_DISPLAY_NAMES[model_key],
                        "dimension": dimension,
                        "category": category,
                        "f1_used": round(f1_value, 8),
                        "raw_f1": round(float(metrics["f1"]), 8),
                        "precision": round(float(metrics["precision"]), 8),
                        "recall": round(float(metrics["recall"]), 8),
                        "tp": int(metrics["tp"]),
                        "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]),
                        "support": int(metrics["support"]),
                        "fallback_to_global_f1": used_fallback,
                    }
                )

    return reliability, csv_rows


def content_reliability_for_candidate(
    model_key: str,
    entity: dict,
    reliability: dict[str, dict],
) -> tuple[float, dict[str, float]]:
    categories = content_categories(entity)
    components: dict[str, float] = {}
    for dimension, category in categories.items():
        components[dimension] = float(
            reliability[model_key][dimension][category]["f1"]
        )
    score = sum(components.values()) / len(components)
    return score, components


def model_candidate_weight(
    model_key: str,
    entity: dict,
    *,
    alpha: float,
    global_weights: dict[str, float],
    reliability: dict[str, dict],
) -> tuple[float, float, dict[str, float]]:
    content_score, components = content_reliability_for_candidate(
        model_key, entity, reliability
    )
    global_score = float(global_weights[model_key])
    weight = alpha * global_score + (1.0 - alpha) * content_score
    return weight, content_score, components


def score_candidates(
    vote_table: dict,
    entity_store: dict,
    *,
    alpha: float,
    global_weights: dict[str, float],
    reliability: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for key, raw_voters in vote_table.items():
        voters = sorted(raw_voters)
        entity = dict(entity_store[key])
        categories = content_categories(entity)
        total_score = 0.0
        voter_details: dict[str, dict] = {}

        for model_key in voters:
            weight, content_score, components = model_candidate_weight(
                model_key,
                entity,
                alpha=alpha,
                global_weights=global_weights,
                reliability=reliability,
            )
            total_score += weight
            voter_details[model_key] = {
                "global_f1": round(float(global_weights[model_key]), 8),
                "content_reliability": round(content_score, 8),
                "cwa_weight": round(weight, 8),
                "components": {k: round(v, 8) for k, v in components.items()},
            }

        rows.append(
            {
                "key": key,
                "entity": entity,
                "voters": voters,
                "vote_count": len(voters),
                "categories": categories,
                "score": round(total_score, 8),
                "voter_details": voter_details,
            }
        )

    rows.sort(key=lambda row: (-float(row["score"]), row["key"]))
    return rows


def sweep_thresholds(
    scored_candidates: list[dict],
    gold_entities: list[dict],
    *,
    alpha: float,
) -> list[dict]:
    gold_keys = {_entity_key(entity) for entity in deduplicate_entities(gold_entities)}
    gold_total = len(gold_keys)

    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in scored_candidates:
        grouped[float(row["score"])].append(row)

    tp = 0
    fp = 0
    accepted = 0
    results: list[dict] = []

    for threshold in sorted(grouped.keys(), reverse=True):
        for row in grouped[threshold]:
            accepted += 1
            if row["key"] in gold_keys:
                tp += 1
            else:
                fp += 1

        fn = gold_total - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        results.append(
            {
                "alpha": round(float(alpha), 4),
                "threshold": round(float(threshold), 8),
                "precision": round(precision, 8),
                "recall": round(recall, 8),
                "f1": round(f1, 8),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "entity_count": accepted,
            }
        )

    return results


def materialise_cwa(
    scored_candidates: list[dict],
    *,
    threshold: float,
    alpha: float,
) -> list[dict]:
    output: list[dict] = []
    for row in scored_candidates:
        if float(row["score"]) + 1e-12 < float(threshold):
            continue
        entity = dict(row["entity"])
        entity["voters"] = list(row["voters"])
        entity["vote_count"] = int(row["vote_count"])
        entity["agreement_score"] = round(
            int(row["vote_count"]) / len(MODEL_KEYS), 6
        )
        entity["cwa_score"] = round(float(row["score"]), 8)
        entity["cwa_alpha"] = round(float(alpha), 4)
        entity["cwa_categories"] = dict(row["categories"])
        entity["cwa_voter_details"] = dict(row["voter_details"])
        output.append(entity)

    output.sort(key=_entity_key)
    return output


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_frozen_wv_config() -> dict:
    path = require_file(
        frozen_config_file(),
        "existing frozen WV development configuration; run candidate_gold --split dev first",
    )
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("fit_split") != "dev":
        raise ValueError("Existing WV configuration was not fitted on development data.")
    return config


def fit_on_development(alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID) -> dict:
    print("Loading existing frozen WV configuration...")
    wv_config = _load_frozen_wv_config()
    global_weights = {key: float(wv_config["weights"][key]) for key in MODEL_KEYS}

    print("Loading BioRED development predictions...")
    model_predictions = load_model_predictions("dev")
    gold_entities = load_jsonl(
        require_file(gold_entities_file("dev"), "BioRED development human gold")
    )

    print("\nLearning content-specific DEV reliability...")
    reliability, reliability_rows = learn_content_reliability(
        gold_entities, model_predictions, global_weights
    )

    for model_key in MODEL_KEYS:
        print(f"\n  {MODEL_DISPLAY_NAMES[model_key]}")
        for dimension in ("entity_type", "span_length", "mention_form"):
            pieces = []
            for category, values in reliability[model_key][dimension].items():
                pieces.append(f"{category}={values['f1']:.4f}")
            print(f"    {dimension}: " + ", ".join(pieces))

    vote_table, entity_store = build_vote_table(model_predictions)
    all_rows: list[dict] = []

    print("\nTuning alpha and CWA threshold on DEV only...")
    for alpha in alpha_grid:
        scored = score_candidates(
            vote_table,
            entity_store,
            alpha=float(alpha),
            global_weights=global_weights,
            reliability=reliability,
        )
        rows = sweep_thresholds(scored, gold_entities, alpha=float(alpha))
        all_rows.extend(rows)
        best_alpha = max(
            rows,
            key=lambda row: (
                float(row["f1"]),
                float(row["precision"]),
                float(row["threshold"]),
            ),
        )
        print(
            f"  alpha={alpha:.1f}  best threshold={best_alpha['threshold']:.8f}  "
            f"P={best_alpha['precision']:.4f} R={best_alpha['recall']:.4f} "
            f"F1={best_alpha['f1']:.4f}"
        )

                                   
                                                          
                                                                     
                          
    best = max(
        all_rows,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["alpha"]),
            float(row["threshold"]),
        ),
    )
    selected_alpha = float(best["alpha"])
    selected_threshold = float(best["threshold"])

    selected_scored = score_candidates(
        vote_table,
        entity_store,
        alpha=selected_alpha,
        global_weights=global_weights,
        reliability=reliability,
    )
    cwa_dev = materialise_cwa(
        selected_scored,
        threshold=selected_threshold,
        alpha=selected_alpha,
    )
                                                                  
    verified = exact_span_metrics(gold_entities, cwa_dev)
    if not math.isclose(float(verified["f1"]), float(best["f1"]), abs_tol=1e-8):
        raise AssertionError("CWA development sweep and materialised output disagree.")

    save_jsonl(cwa_dev, cwa_pseudo_gold_file("dev"))
    output_dir = cwa_results_dir("dev")
    _write_csv(reliability_rows, output_dir / "content_reliability.csv")
    _write_csv(all_rows, output_dir / "alpha_threshold_selection.csv")

    config = {
        "schema_version": CWA_SCHEMA_VERSION,
        "method": "Content-Aware Weighted Voting",
        "fit_split": "dev",
        "primary_metric": "exact_character_span_and_label_micro_f1",
        "global_weight_source": str(frozen_config_file()),
        "global_weights": global_weights,
        "content_dimensions": ["entity_type", "span_length", "mention_form"],
        "span_length_rule": "one whitespace-delimited token vs more than one",
        "mention_form_rule": (
            "abbreviation_alphanumeric if letters+digits OR 2-10 alphabetic characters "
            "are all uppercase; otherwise ordinary"
        ),
        "content_reliability_definition": (
            "arithmetic mean of DEV exact-span F1 for the candidate's entity type, "
            "span-length category, and mention-form category"
        ),
        "candidate_weight_formula": (
            "w_i(e) = alpha * global_dev_f1_i + (1-alpha) * content_reliability_i(e)"
        ),
        "zero_support_fallback": "use the model's frozen global DEV F1",
        "alpha_grid": list(alpha_grid),
        "selection_rule": (
            "highest DEV F1, then highest precision, then higher alpha, then higher threshold"
        ),
        "selected_alpha": selected_alpha,
        "selected_threshold": selected_threshold,
        "selected_dev_metrics": best,
        "content_reliability": reliability,
        "dev_candidate_count": len(vote_table),
        "dev_cwa_entity_count": len(cwa_dev),
    }
    save_json(config, cwa_config_file())

    print("\nFrozen CWA development configuration:")
    print(f"  alpha:      {selected_alpha:.4f}")
    print(f"  threshold:  {selected_threshold:.8f}")
    print(f"  DEV P/R/F1: {verified['precision']:.4f}/{verified['recall']:.4f}/{verified['f1']:.4f}")
    print(f"  config:     {cwa_config_file()}")
    print(f"  CWA dev:    {cwa_pseudo_gold_file('dev')}")
    return config


def apply_to_test(config_path: Path) -> None:
    require_file(config_path, "frozen CWA development configuration")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if config.get("fit_split") != "dev":
        raise ValueError("CWA configuration was not fitted on development data.")

    alpha = float(config["selected_alpha"])
    threshold = float(config["selected_threshold"])
    global_weights = {
        model: float(config["global_weights"][model]) for model in MODEL_KEYS
    }
    reliability = config["content_reliability"]

                                                                        
    print("Loading BioRED test predictions only...")
    model_predictions = load_model_predictions("test")
    vote_table, entity_store = build_vote_table(model_predictions)
    scored = score_candidates(
        vote_table,
        entity_store,
        alpha=alpha,
        global_weights=global_weights,
        reliability=reliability,
    )
    cwa_test = materialise_cwa(scored, threshold=threshold, alpha=alpha)
    output_path = cwa_pseudo_gold_file("test")
    save_jsonl(cwa_test, output_path)

    application = {
        "method": "Content-Aware Weighted Voting",
        "applied_split": "test",
        "source_config": str(config_path),
        "selected_alpha": alpha,
        "selected_threshold": threshold,
        "candidate_count": len(vote_table),
        "cwa_entity_count": len(cwa_test),
        "test_gold_was_not_read": True,
    }
    save_json(application, cwa_results_dir("test") / "cwa_application.json")

    print("\nApplied frozen CWA settings to TEST predictions.")
    print(f"  alpha:      {alpha:.4f}")
    print(f"  threshold:  {threshold:.8f}")
    print(f"  CWA test:   {output_path} ({len(cwa_test)} entities)")
    print("  Test human gold was NOT accessed during construction.")


def _parse_alpha_grid(value: str | None) -> tuple[float, ...]:
    if value is None:
        return DEFAULT_ALPHA_GRID
    numbers = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not numbers:
        raise ValueError("Alpha grid cannot be empty.")
    for number in numbers:
        if number <= 0.0 or number >= 1.0:
            raise ValueError("Every alpha must be greater than 0 and less than 1.")
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description="BioRED Content-Aware Weighted Voting.")
    parser.add_argument("--split", required=True, choices=["dev", "development", "test"])
    parser.add_argument(
        "--config",
        type=Path,
        default=cwa_config_file(),
        help="Frozen CWA DEV config used for --split test.",
    )
    parser.add_argument(
        "--alpha-grid",
        default=None,
        help="Optional comma-separated DEV-only alpha grid. Default: 0.1,0.2,...,0.9",
    )
    args = parser.parse_args()

    split = normalize_split(args.split, allow_train=False)
    if split == "dev":
        fit_on_development(_parse_alpha_grid(args.alpha_grid))
    else:
        apply_to_test(args.config)


if __name__ == "__main__":
    main()
