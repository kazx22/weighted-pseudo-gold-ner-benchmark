"""Compare official BioRED with the BC5CDR-overlap-excluded sensitivity run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {
    "official": ROOT / "results" / "biored",
    "overlap_excluded": ROOT / "results" / "biored_overlap_excluded",
}
OUTPUT_DIR = ROOT / "results" / "biored_sensitivity_comparison"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _find(rows: list[dict], system: str) -> dict:
    for row in rows:
        if str(row.get("system")) == system:
            return row
    raise KeyError(f"Could not find {system!r}")


def _summary(variant: str, root: Path, audit: dict) -> dict:
    evaluation = _load(root / "test" / "evaluation_results.json")
    med = _load(root / "test" / "medgemma_evaluation.json")
    controls = _load(root / "test" / "method_controls" / "method_controls.json")
    config = _load(root / "dev" / "frozen_pseudo_gold_config.json")
    budget_payload = _load(root / "test" / "development_gold_sensitivity" / "development_gold_sensitivity.json")
    budget = {str(int(item["percent"])): float(item["mean_test_f1"]) for item in budget_payload["summary"]}

    pseudo = evaluation["pseudo_gold_vs_human"]
    models = evaluation["models_vs_human"]
    med_rows = med["results"]
    overlap_split = (audit.get("splits") or {}).get("test", {})

    row = {
        "variant": variant,
        "dev_documents": None,
        "test_documents": None,
        "removed_test_overlap_documents": 0,
        "selected_weighted_threshold": float(config["selected_threshold"]),
        "majority_f1": float(_find(pseudo, "Majority Pseudo-Gold")["f1"]),
        "weighted_f1": float(_find(pseudo, "Weighted Pseudo-Gold")["f1"]),
        "scispacy_f1": float(_find(models, "scispaCy")["f1"]),
        "tuned_unweighted_k": int(controls["tuned_unweighted"]["selected_k"]),
        "tuned_unweighted_f1": float(controls["tuned_unweighted"]["test_metrics"]["f1"]),
        "weighted_minus_tuned_unweighted_f1": float(
            controls["tuned_unweighted"]["weighted_minus_tuned_unweighted_f1"]
        ),
        "weighted_vs_tuned_unweighted_p": float(
            controls["tuned_unweighted"]["paired_bootstrap_raw_p"]
        ),
        "medgemma_zero_shot_f1": float(_find(med_rows, "MedGemma Zero-Shot")["f1"]),
        "hybrid_f1": float(_find(med_rows, "Weighted + MedGemma Tie-Breaker")["f1"]),
        "hybrid_disease_f1": float(
            _find(med_rows, "Weighted + MedGemma Tie-Breaker")["per_label"]["DISEASE"]["f1-score"]
        ),
        "hybrid_chemical_f1": float(
            _find(med_rows, "Weighted + MedGemma Tie-Breaker")["per_label"]["CHEMICAL"]["f1-score"]
        ),
        "routing_rate": float((med.get("hybrid_routing") or {}).get("routing_rate_over_all_candidates", 0.0)),
        "zero_shot_request_minutes": float((med.get("zero_shot_diagnostics") or {}).get("summed_request_seconds", 0.0)) / 60.0,
        "selective_request_minutes": float((med.get("hybrid_routing") or {}).get("tie_break_request_seconds", 0.0)) / 60.0,
        "budget_25_mean_test_f1": budget["25"],
        "budget_50_mean_test_f1": budget["50"],
        "budget_75_mean_test_f1": budget["75"],
        "budget_100_test_f1": budget["100"],
    }

    if variant == "overlap_excluded":
        row["dev_documents"] = int(audit["splits"]["dev"]["retained_documents"])
        row["test_documents"] = int(audit["splits"]["test"]["retained_documents"])
        row["removed_test_overlap_documents"] = int(overlap_split["overlap_documents"])
    else:
        row["dev_documents"] = int(audit["splits"]["dev"]["original_documents"])
        row["test_documents"] = int(audit["splits"]["test"]["original_documents"])
    return row


def main() -> None:
    audit = _load(ROOT / "results" / "biored_overlap_excluded" / "overlap_audit.json")
    rows = [_summary(name, root, audit) for name, root in VARIANTS.items()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "official_vs_overlap_excluded.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "design": "official BioRED primary analysis plus BC5CDR-overlap-excluded sensitivity analysis",
        "overlap_rule": audit["rule"],
        "variants": rows,
    }
    (OUTPUT_DIR / "official_vs_overlap_excluded.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    official, clean = rows
    lines = [
        "# BioRED overlap sensitivity comparison",
        "",
        f"Official BioRED test: {official['test_documents']} documents.",
        f"Overlap-excluded BioRED test: {clean['test_documents']} documents "
        f"({clean['removed_test_overlap_documents']} BC5CDR-overlapping documents removed).",
        "",
        "| Metric | Official | Overlap-excluded |",
        "|---|---:|---:|",
    ]
    metrics = [
        ("Majority F1", "majority_f1"),
        ("Weighted F1", "weighted_f1"),
        ("scispaCy F1", "scispacy_f1"),
        ("Tuned unweighted F1", "tuned_unweighted_f1"),
        ("MedGemma zero-shot F1", "medgemma_zero_shot_f1"),
        ("Weighted + MedGemma F1", "hybrid_f1"),
        ("25% dev-gold mean weighted F1", "budget_25_mean_test_f1"),
        ("50% dev-gold mean weighted F1", "budget_50_mean_test_f1"),
        ("75% dev-gold mean weighted F1", "budget_75_mean_test_f1"),
        ("100% dev-gold weighted F1", "budget_100_test_f1"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {official[key]:.4f} | {clean[key]:.4f} |")
    lines += [
        "",
        "The overlap-excluded branch is recalibrated independently on its cleaned development split; "
        "official BioRED results are retained rather than replaced.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Sensitivity comparison: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
