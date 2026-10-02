
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.content_aware_voting import (
    content_categories,
    cwa_pseudo_gold_file,
    cwa_results_dir,
)
from src.experiment_config import (
    docs_file,
    figures_dir,
    gold_entities_file,
    normalize_split,
    pseudo_gold_file,
    require_file,
    save_json,
)
from src.utils import (
    deduplicate_entities,
    exact_span_metrics,
    group_by_row,
    load_jsonl,
    metrics_from_counts,
    per_document_exact_counts,
)

N_BOOTSTRAP = 1000
SEED = 42


def filter_by_category(
    entities: list[dict], dimension: str, category: str
) -> list[dict]:
    output: list[dict] = []
    for entity in deduplicate_entities(entities):
        if content_categories(entity)[dimension] == category:
            output.append(entity)
    return output


def evaluate_method(gold: list[dict], predicted: list[dict]) -> dict:
    return exact_span_metrics(gold, predicted)


def evaluate_subgroups(gold: list[dict], predicted: list[dict]) -> dict[str, dict]:
    specification = {
        "entity_type": ("DISEASE", "CHEMICAL"),
        "span_length": ("single_token", "multi_token"),
        "mention_form": ("ordinary", "abbreviation_alphanumeric"),
    }
    output: dict[str, dict] = {}
    for dimension, categories in specification.items():
        output[dimension] = {}
        for category in categories:
            gold_subset = filter_by_category(gold, dimension, category)
            pred_subset = filter_by_category(predicted, dimension, category)
            output[dimension][category] = exact_span_metrics(gold_subset, pred_subset)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_overall(overall: dict[str, dict], output_file: Path) -> None:
    methods = ["MV", "WV", "CWA"]
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for index, metric in enumerate(metrics):
        values = [float(overall[method][metric]) for method in methods]
        ax.bar(x + (index - 1) * width, values, width, label=metric.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Exact-span score")
    ax.set_title("MV vs WV vs CWA on BC5CDR")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_subgroup_f1(subgroups: dict[str, dict], output_file: Path) -> None:
    methods = ["MV", "WV", "CWA"]
    categories = [
        ("entity_type", "DISEASE", "Disease"),
        ("entity_type", "CHEMICAL", "Chemical"),
        ("span_length", "single_token", "Single-token"),
        ("span_length", "multi_token", "Multi-token"),
        ("mention_form", "ordinary", "Ordinary"),
        ("mention_form", "abbreviation_alphanumeric", "Abbrev./alnum."),
    ]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for index, method in enumerate(methods):
        values = [
            float(subgroups[method][dimension][category]["f1"])
            for dimension, category, _label in categories
        ]
        ax.bar(x + (index - 1) * width, values, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label for _dimension, _category, label in categories],
        rotation=20,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Exact-span F1")
    ax.set_title("Content-Specific F1: MV vs WV vs CWA")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _aggregate_metrics(counts: list[dict], indices: np.ndarray) -> dict:
    tp = fp = fn = 0
    for index in indices:
        row = counts[int(index)]
        tp += int(row["tp"])
        fp += int(row["fp"])
        fn += int(row["fn"])
    return metrics_from_counts(tp, fp, fn)


def _paired_bootstrap_p_value(
    first_bootstrap: np.ndarray,
    second_bootstrap: np.ndarray,
) -> float:
    if len(first_bootstrap) != len(second_bootstrap):
        raise ValueError("Paired bootstrap arrays must have equal length.")

    n = len(first_bootstrap)
    difference = first_bootstrap - second_bootstrap
    lower_tail = (np.count_nonzero(difference <= 0) + 1) / (n + 1)
    upper_tail = (np.count_nonzero(difference >= 0) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


def _holm_adjust(raw_p_values: list[float]) -> list[float]:
    number = len(raw_p_values)
    if number == 0:
        return []

    order = sorted(range(number), key=lambda index: raw_p_values[index])
    adjusted = [0.0] * number
    running_max = 0.0

    for rank, original_index in enumerate(order):
        candidate = min(
            1.0,
            (number - rank) * float(raw_p_values[original_index]),
        )
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted


def paired_document_bootstrap(
    docs: list[dict],
    gold: list[dict],
    methods: dict[str, list[dict]],
    *,
    resamples: int,
    seed: int,
) -> dict:
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")

    gold_by_row = group_by_row(gold)
    counts = {
        method: per_document_exact_counts(
            docs,
            gold_by_row,
            group_by_row(predicted),
        )
        for method, predicted in methods.items()
    }

    n_docs = len(docs)
    if n_docs == 0:
        raise ValueError("No documents available for bootstrap.")

    full_indices = np.arange(n_docs)
    observed = {
        method: _aggregate_metrics(method_counts, full_indices)
        for method, method_counts in counts.items()
    }

    rng = np.random.default_rng(seed)
    samples = {
        method: np.zeros(resamples, dtype=float)
        for method in methods
    }

    for iteration in range(resamples):
        sampled_indices = rng.integers(0, n_docs, size=n_docs)
        for method, method_counts in counts.items():
            samples[method][iteration] = float(
                _aggregate_metrics(method_counts, sampled_indices)["f1"]
            )

        if (iteration + 1) % 200 == 0:
            print(f"Bootstrap: completed {iteration + 1}/{resamples} resamples")

    confidence_intervals: dict[str, dict] = {}
    for method in ("MV", "WV", "CWA"):
        low, high = np.percentile(samples[method], [2.5, 97.5])
        confidence_intervals[method] = {
            "observed_f1": float(observed[method]["f1"]),
            "ci_low": float(low),
            "ci_high": float(high),
        }

    planned = [
        ("CWA", "WV"),
        ("CWA", "MV"),
    ]
    comparison_rows: list[dict] = []
    raw_p_values: list[float] = []

    for first, second in planned:
        raw_p = _paired_bootstrap_p_value(samples[first], samples[second])
        raw_p_values.append(raw_p)
        comparison_rows.append(
            {
                "comparison": f"{first}_vs_{second}",
                "method_1": first,
                "method_2": second,
                "f1_method_1": float(observed[first]["f1"]),
                "f1_method_2": float(observed[second]["f1"]),
                "f1_difference": float(
                    observed[first]["f1"] - observed[second]["f1"]
                ),
                "raw_p": raw_p,
            }
        )

    adjusted = _holm_adjust(raw_p_values)
    for row, holm_p in zip(comparison_rows, adjusted):
        row["holm_p"] = float(holm_p)
        row["significant_holm_0.05"] = bool(holm_p < 0.05)

    return {
        "method": "paired_document_level_bootstrap",
        "primary_metric": "exact_character_span_and_label_micro_f1",
        "resamples": int(resamples),
        "seed": int(seed),
        "family": [
            "CWA_vs_WV",
            "CWA_vs_MV",
        ],
        "multiple_testing_correction": "Holm",
        "confidence_intervals_95": confidence_intervals,
        "comparisons": comparison_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MV, WV, and CWA on BC5CDR.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "development", "test"],
    )
    parser.add_argument("--resamples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split = normalize_split(args.split, allow_train=False)

    human_gold = load_jsonl(
        require_file(gold_entities_file(split), f"{split} human gold entities")
    )
    methods = {
        "MV": load_jsonl(
            require_file(
                pseudo_gold_file("majority", split),
                f"{split} majority pseudo-gold",
            )
        ),
        "WV": load_jsonl(
            require_file(
                pseudo_gold_file("weighted", split),
                f"{split} weighted pseudo-gold",
            )
        ),
        "CWA": load_jsonl(
            require_file(
                cwa_pseudo_gold_file(split),
                f"{split} CWA pseudo-gold",
            )
        ),
    }

    overall: dict[str, dict] = {}
    subgroups: dict[str, dict] = {}
    overall_rows: list[dict] = []
    subgroup_rows: list[dict] = []

    print("\n" + "=" * 72)
    print(f"{split.upper()} — MV -> WV -> CWA")
    print("=" * 72)

    for method, predicted in methods.items():
        metrics = evaluate_method(human_gold, predicted)
        overall[method] = metrics
        subgroups[method] = evaluate_subgroups(human_gold, predicted)
        print(
            f"{method:<3}  P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
        )
        overall_rows.append({"method": method, **metrics})

        for dimension, category_values in subgroups[method].items():
            for category, category_metrics in category_values.items():
                subgroup_rows.append(
                    {
                        "method": method,
                        "dimension": dimension,
                        "category": category,
                        **category_metrics,
                    }
                )

    print("\nEntity-type F1:")
    for method in ("MV", "WV", "CWA"):
        values = subgroups[method]["entity_type"]
        print(
            f"  {method}: DISEASE={values['DISEASE']['f1']:.4f}  "
            f"CHEMICAL={values['CHEMICAL']['f1']:.4f}"
        )

    print("\nSpan-length F1:")
    for method in ("MV", "WV", "CWA"):
        values = subgroups[method]["span_length"]
        print(
            f"  {method}: single={values['single_token']['f1']:.4f}  "
            f"multi={values['multi_token']['f1']:.4f}"
        )

    print("\nMention-form F1:")
    for method in ("MV", "WV", "CWA"):
        values = subgroups[method]["mention_form"]
        print(
            f"  {method}: ordinary={values['ordinary']['f1']:.4f}  "
            f"abbrev/alnum={values['abbreviation_alphanumeric']['f1']:.4f}"
        )

    bootstrap = None
    bootstrap_ci_rows: list[dict] = []
    bootstrap_comparison_rows: list[dict] = []

                                                             
                                                                            
                               
    # Paired bootstrap is reported only on held-out test data.
    if split == "test":
        docs = load_jsonl(require_file(docs_file(split), f"{split} documents"))
        print("\n" + "=" * 72)
        print("PAIRED DOCUMENT-LEVEL BOOTSTRAP")
        print("=" * 72)

        bootstrap = paired_document_bootstrap(
            docs,
            human_gold,
            methods,
            resamples=args.resamples,
            seed=args.seed,
        )

        print("\n95% bootstrap confidence intervals:")
        for method in ("MV", "WV", "CWA"):
            row = bootstrap["confidence_intervals_95"][method]
            print(
                f"  {method}: F1={row['observed_f1']:.4f} "
                f"CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
            bootstrap_ci_rows.append(
                {
                    "method": method,
                    **row,
                }
            )

        print("\nPlanned paired comparisons:")
        for row in bootstrap["comparisons"]:
            print(
                f"  {row['method_1']} vs {row['method_2']}: "
                f"delta F1={row['f1_difference']:+.4f}  "
                f"raw p={row['raw_p']:.4f}  "
                f"Holm p={row['holm_p']:.4f}"
            )
            bootstrap_comparison_rows.append(dict(row))

    output_dir = cwa_results_dir(split)
    payload = {
        "split": split,
        "comparison_order": ["MV", "WV", "CWA"],
        "primary_metric": "exact_character_span_and_label_micro_f1",
        "overall": overall,
        "subgroups": subgroups,
        "bootstrap": bootstrap,
    }
    save_json(payload, output_dir / "cwa_evaluation.json")

    if bootstrap is not None:
        save_json(bootstrap, output_dir / "cwa_bootstrap.json")

    _write_csv(output_dir / "overall_comparison.csv", overall_rows)
    _write_csv(output_dir / "subgroup_comparison.csv", subgroup_rows)
    _write_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        bootstrap_ci_rows,
    )
    _write_csv(
        output_dir / "bootstrap_comparisons.csv",
        bootstrap_comparison_rows,
    )

    figure_dir = figures_dir(split)
    _plot_overall(overall, figure_dir / "mv_wv_cwa_comparison.png")
    _plot_subgroup_f1(
        subgroups,
        figure_dir / "mv_wv_cwa_subgroup_f1.png",
    )

    print(f"\nSaved CWA evaluation to {output_dir}")
    if bootstrap is not None:
        print(f"Saved bootstrap: {output_dir / 'cwa_bootstrap.json'}")
    print(f"Saved figure: {figure_dir / 'mv_wv_cwa_comparison.png'}")
    print(f"Saved figure: {figure_dir / 'mv_wv_cwa_subgroup_f1.png'}")


if __name__ == "__main__":
    main()
