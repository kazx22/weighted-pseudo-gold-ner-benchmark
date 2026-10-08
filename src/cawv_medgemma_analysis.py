from __future__ import annotations

import argparse
import csv
import importlib

import numpy as np

from src.bootstrap_significance import paired_bootstrap_p_value
from src.cawv_medgemma_hybrid import dataset_parts, output_file
from src.utils import exact_span_metrics, group_by_row, load_jsonl, metrics_from_counts, per_document_exact_counts


def aggregate(counts: list[dict], indices: np.ndarray) -> dict:
    tp = fp = fn = 0
    for index in indices:
        row = counts[int(index)]
        tp += int(row["tp"])
        fp += int(row["fp"])
        fn += int(row["fn"])
    return metrics_from_counts(tp, fp, fn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "biored"], required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg, cawv, _, suffix = dataset_parts(args.dataset)
    gold = load_jsonl(cfg.require_file(cfg.gold_entities_file(args.split), "human gold"))
    baseline = load_jsonl(cfg.require_file(cawv.cawv_pseudo_gold_file(args.split), "CAWV output"))
    hybrid = load_jsonl(cfg.require_file(output_file(cfg, args.split, suffix), "CAWV + MedGemma output"))

    baseline_metrics = exact_span_metrics(gold, baseline)
    hybrid_metrics = exact_span_metrics(gold, hybrid)
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "primary_metric": "exact_character_span_and_label_micro_f1",
        "cawv": baseline_metrics,
        "cawv_plus_medgemma": hybrid_metrics,
        "f1_difference": float(hybrid_metrics["f1"] - baseline_metrics["f1"]),
    }

    if args.split == "test":
        docs = load_jsonl(cfg.require_file(cfg.docs_file("test"), "test documents"))
        gold_by_row = group_by_row(gold)
        counts = {
            "CAWV": per_document_exact_counts(docs, gold_by_row, group_by_row(baseline)),
            "CAWV+MedGemma": per_document_exact_counts(docs, gold_by_row, group_by_row(hybrid)),
        }
        full = np.arange(len(docs))
        observed = {key: aggregate(value, full) for key, value in counts.items()}
        rng = np.random.default_rng(args.seed)
        samples = {key: np.zeros(args.resamples, dtype=float) for key in counts}
        for i in range(args.resamples):
            indices = rng.integers(0, len(docs), size=len(docs))
            for key in counts:
                samples[key][i] = float(aggregate(counts[key], indices)["f1"])
        p = paired_bootstrap_p_value(samples["CAWV+MedGemma"], samples["CAWV"], resamples=args.resamples)
        ci = {}
        for key in samples:
            low, high = np.percentile(samples[key], [2.5, 97.5])
            ci[key] = {"observed_f1": float(observed[key]["f1"]), "ci_low": float(low), "ci_high": float(high)}
        payload["bootstrap"] = {
            "resamples": args.resamples,
            "seed": args.seed,
            "confidence_intervals_95": ci,
            "comparison": {
                "method_1": "CAWV+MedGemma",
                "method_2": "CAWV",
                "f1_difference": float(observed["CAWV+MedGemma"]["f1"] - observed["CAWV"]["f1"]),
                "two_sided_p": float(p),
                "significant_0.05": bool(p < 0.05),
            },
        }

    result_dir = cfg.results_dir(args.split) / "medgemma_cawv_hybrid"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(payload, result_dir / "cawv_medgemma_analysis.json")

    rows = [
        {"system": "CAWV", **{k: baseline_metrics[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}},
        {"system": "CAWV + MedGemma", **{k: hybrid_metrics[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}},
    ]
    with (result_dir / "cawv_medgemma_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CAWV F1={baseline_metrics['f1']:.4f}")
    print(f"CAWV + MedGemma F1={hybrid_metrics['f1']:.4f}")
    print(f"Delta F1={payload['f1_difference']:+.4f}")
    if args.split == "test":
        print(f"Paired bootstrap p={payload['bootstrap']['comparison']['two_sided_p']:.4f}")


if __name__ == "__main__":
    main()
