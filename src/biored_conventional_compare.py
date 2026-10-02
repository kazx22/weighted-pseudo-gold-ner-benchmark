
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


def _budget_map(root: Path) -> dict[str, float]:
    payload = _load(
        root / "test" / "development_gold_sensitivity" / "development_gold_sensitivity.json"
    )
    return {
        str(int(row["percent"])): float(row["mean_test_f1"])
        for row in payload["summary"]
    }


def _summary(variant: str, root: Path, audit: dict) -> dict:
    evaluation = _load(root / "test" / "evaluation_results.json")
    controls = _load(root / "test" / "method_controls" / "method_controls.json")
    config = _load(root / "dev" / "frozen_pseudo_gold_config.json")
    budget = _budget_map(root)

    pseudo = evaluation["pseudo_gold_vs_human"]
    models = evaluation["models_vs_human"]
    split_info = audit["splits"]["test"]

    row = {
        "variant": variant,
        "dev_documents": int(audit["splits"]["dev"]["original_documents"]),
        "test_documents": int(split_info["original_documents"]),
        "removed_dev_overlap_documents": 0,
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
        "budget_25_mean_test_f1": budget["25"],
        "budget_50_mean_test_f1": budget["50"],
        "budget_75_mean_test_f1": budget["75"],
        "budget_100_test_f1": budget["100"],
    }
    if variant == "overlap_excluded":
        row["dev_documents"] = int(audit["splits"]["dev"]["retained_documents"])
        row["test_documents"] = int(audit["splits"]["test"]["retained_documents"])
        row["removed_dev_overlap_documents"] = int(audit["splits"]["dev"]["overlap_documents"])
        row["removed_test_overlap_documents"] = int(audit["splits"]["test"]["overlap_documents"])
    return row


def main() -> None:
    audit = _load(ROOT / "results" / "biored_overlap_excluded" / "overlap_audit.json")
    rows = [_summary(name, root, audit) for name, root in VARIANTS.items()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "conventional_official_vs_overlap_excluded.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "design": "official BioRED plus independently recalibrated BC5CDR-overlap-excluded sensitivity analysis",
        "overlap_rule": audit["rule"],
        "medgemma_included": False,
        "variants": rows,
    }
    (OUTPUT_DIR / "conventional_official_vs_overlap_excluded.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    official, clean = rows
    lines = [
        "# BioRED conventional overlap-sensitivity comparison",
        "",
        f"Official: {official['dev_documents']} dev / {official['test_documents']} test documents.",
        f"Overlap-excluded: {clean['dev_documents']} dev / {clean['test_documents']} test documents.",
        f"Removed: {clean['removed_dev_overlap_documents']} dev and {clean['removed_test_overlap_documents']} test documents shared with BC5CDR.",
        "",
        "| Metric | Official | Overlap-excluded |",
        "|---|---:|---:|",
        f"| Majority F1 | {official['majority_f1']:.4f} | {clean['majority_f1']:.4f} |",
        f"| Weighted F1 | {official['weighted_f1']:.4f} | {clean['weighted_f1']:.4f} |",
        f"| scispaCy F1 | {official['scispacy_f1']:.4f} | {clean['scispacy_f1']:.4f} |",
        f"| Tuned unweighted F1 | {official['tuned_unweighted_f1']:.4f} | {clean['tuned_unweighted_f1']:.4f} |",
        f"| 25% dev-gold mean weighted F1 | {official['budget_25_mean_test_f1']:.4f} | {clean['budget_25_mean_test_f1']:.4f} |",
        f"| 50% dev-gold mean weighted F1 | {official['budget_50_mean_test_f1']:.4f} | {clean['budget_50_mean_test_f1']:.4f} |",
        f"| 75% dev-gold mean weighted F1 | {official['budget_75_mean_test_f1']:.4f} | {clean['budget_75_mean_test_f1']:.4f} |",
        f"| 100% dev-gold weighted F1 | {official['budget_100_test_f1']:.4f} | {clean['budget_100_test_f1']:.4f} |",
        "",
        "The overlap-excluded branch has its own development-fitted weights and threshold; it does not reuse the official BioRED calibration.",
    ]
    (OUTPUT_DIR / "CONVENTIONAL_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Conventional sensitivity comparison: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
