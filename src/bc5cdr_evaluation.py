"""Split-aware BC5CDR evaluation with exact character-span primary metrics.

Primary precision, recall and F1 require an exact match on:
(row_id, start_char, end_char, label).
BIO labels are generated only for secondary token-level confusion matrices.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.experiment_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    docs_file,
    figures_dir,
    gold_entities_file,
    load_json,
    normalize_split,
    prediction_file,
    pseudo_gold_file,
    require_file,
    results_dir,
    runtime_file,
    save_json,
)
from src.graph import (
    add_human_comparison_result,
    add_model_result_candidate,
    add_model_result_human,
    plot_all,
    reset_results,
    set_figure_dir,
)
from src.utils import (
    exact_per_label_report,
    exact_span_metrics,
    group_by_row,
    load_jsonl,
    span_to_bio,
)


def build_flat_bio(
    docs: list[dict],
    reference_by_row: dict[int, list[dict]],
    prediction_by_row: dict[int, list[dict]],
) -> tuple[list[str], list[str]]:
    flat_true: list[str] = []
    flat_pred: list[str] = []

    for doc in docs:
        row_id = int(doc["row_id"])
        text = str(doc["full_text"])
        true_tokens, true_labels = span_to_bio(text, reference_by_row.get(row_id, []))
        pred_tokens, pred_labels = span_to_bio(text, prediction_by_row.get(row_id, []))
        if true_tokens != pred_tokens or len(true_labels) != len(pred_labels):
            raise ValueError(f"BIO token alignment failed for row_id {row_id}")
        flat_true.extend(true_labels)
        flat_pred.extend(pred_labels)

    return flat_true, flat_pred


def print_metrics(name: str, metrics: dict, report: dict) -> None:
    print(f"\n{name}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    for label in ("DISEASE", "CHEMICAL"):
        item = report[label]
        print(
            f"  {label:<8} P={item['precision']:.4f} "
            f"R={item['recall']:.4f} F1={item['f1-score']:.4f} "
            f"support={item['support']}"
        )


def evaluate_source(
    *,
    name: str,
    docs: list[dict],
    reference_entities: list[dict],
    predicted_entities: list[dict],
    runtime: float | None,
    graph_group: str,
) -> dict:
    metrics = exact_span_metrics(reference_entities, predicted_entities)
    report = exact_per_label_report(reference_entities, predicted_entities)
    print_metrics(name, metrics, report)

    reference_by_row = group_by_row(reference_entities)
    prediction_by_row = group_by_row(predicted_entities)
    flat_true, flat_pred = build_flat_bio(docs, reference_by_row, prediction_by_row)

    graph_args = (
        name,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        report,
        runtime,
    )
    if graph_group == "human_model":
        add_model_result_human(*graph_args, y_true=flat_true, y_pred=flat_pred)
    elif graph_group == "pseudo_model":
        add_model_result_candidate(*graph_args, y_true=flat_true, y_pred=flat_pred)
    elif graph_group == "pseudo_human":
        add_human_comparison_result(*graph_args, y_true=flat_true, y_pred=flat_pred)
    else:
        raise ValueError(f"Unknown graph group: {graph_group}")

    return {
        "name": name,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "runtime_seconds_per_document": runtime,
        "per_label": report,
    }


def write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "comparison",
        "name",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "runtime_seconds_per_document",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one BC5CDR split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
    )
    parser.add_argument(
        "--skip-model-vs-pseudo",
        action="store_true",
        help="Skip secondary model-to-pseudo-gold agreement results.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    reset_results()
    set_figure_dir(figures_dir(split))

    docs = load_jsonl(require_file(docs_file(split), "parsed documents"))
    human_gold = load_jsonl(require_file(gold_entities_file(split), "human gold entities"))
    majority_gold = load_jsonl(
        require_file(pseudo_gold_file("majority", split), "majority pseudo-gold")
    )
    weighted_gold = load_jsonl(
        require_file(pseudo_gold_file("weighted", split), "weighted pseudo-gold")
    )

    runtimes = load_json(runtime_file(split), default={}) or {}
    predictions: dict[str, list[dict]] = {}
    for model_key in MODEL_KEYS:
        predictions[model_key] = load_jsonl(
            require_file(
                prediction_file(model_key, split),
                f"{MODEL_DISPLAY_NAMES[model_key]} predictions",
            )
        )

    all_rows: list[dict] = []
    detailed: dict[str, list[dict]] = {
        "pseudo_gold_vs_human": [],
        "models_vs_human": [],
        "models_vs_majority_pseudo_gold": [],
        "models_vs_weighted_pseudo_gold": [],
    }

    print("\n" + "=" * 72)
    print(f"{split.upper()} — PSEUDO-GOLD VS HUMAN GOLD (EXACT SPAN + LABEL)")
    print("=" * 72)
    for name, entities in (
        ("Majority Pseudo-Gold", majority_gold),
        ("Weighted Pseudo-Gold", weighted_gold),
    ):
        row = evaluate_source(
            name=name,
            docs=docs,
            reference_entities=human_gold,
            predicted_entities=entities,
            runtime=None,
            graph_group="pseudo_human",
        )
        row["comparison"] = "pseudo_gold_vs_human"
        detailed["pseudo_gold_vs_human"].append(row)
        all_rows.append(row)

    print("\n" + "=" * 72)
    print(f"{split.upper()} — MODELS VS HUMAN GOLD (EXACT SPAN + LABEL)")
    print("=" * 72)
    for model_key in MODEL_KEYS:
        display = MODEL_DISPLAY_NAMES[model_key]
        runtime = (runtimes.get(model_key) or {}).get("average_seconds_per_document")
        row = evaluate_source(
            name=display,
            docs=docs,
            reference_entities=human_gold,
            predicted_entities=predictions[model_key],
            runtime=runtime,
            graph_group="human_model",
        )
        row["comparison"] = "models_vs_human"
        detailed["models_vs_human"].append(row)
        all_rows.append(row)

    if not args.skip_model_vs_pseudo:
        for reference_name, reference_entities, key in (
            ("Majority Pseudo-Gold", majority_gold, "models_vs_majority_pseudo_gold"),
            ("Weighted Pseudo-Gold", weighted_gold, "models_vs_weighted_pseudo_gold"),
        ):
            print("\n" + "=" * 72)
            print(f"{split.upper()} — MODELS VS {reference_name.upper()}")
            print("=" * 72)
            for model_key in MODEL_KEYS:
                display = MODEL_DISPLAY_NAMES[model_key]
                runtime = (runtimes.get(model_key) or {}).get(
                    "average_seconds_per_document"
                )
                row = evaluate_source(
                    name=f"{display} vs {reference_name}",
                    docs=docs,
                    reference_entities=reference_entities,
                    predicted_entities=predictions[model_key],
                    runtime=runtime,
                    graph_group="pseudo_model",
                )
                row["comparison"] = key
                detailed[key].append(row)
                all_rows.append(row)

    output_dir = results_dir(split)
    save_json(
        {
            "split": split,
            "primary_metric": "exact_character_span_and_label",
            "results": detailed,
        },
        output_dir / "evaluation_results.json",
    )
    write_summary_csv(all_rows, output_dir / "evaluation_summary.csv")
    plot_all()

    print(f"\nSaved results to {output_dir}")
    print(f"Saved figures to {figures_dir(split)}")


if __name__ == "__main__":
    main()
