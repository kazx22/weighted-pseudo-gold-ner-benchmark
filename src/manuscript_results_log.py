from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from src.pipeline_common import ROOT

OUTPUT = ROOT / "results" / "MANUSCRIPT_RESULTS.log"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def find_row(rows: list[dict], name: str) -> dict | None:
    for row in rows:
        candidate = row.get("name", row.get("system"))
        if candidate == name:
            return row
    return None


def add_main_dataset(lines: list[str], label: str, root: Path) -> None:
    evaluation = load_json(root / "test" / "evaluation_results.json")
    cawv = load_json(root / "test" / "cawv" / "cawv_evaluation.json")
    if evaluation is None:
        lines += [f"[{label}]", "NOT AVAILABLE", ""]
        return

    if "results" in evaluation:
        pseudo = evaluation["results"]["pseudo_gold_vs_human"]
        models = evaluation["results"]["models_vs_human"]
    else:
        pseudo = evaluation["pseudo_gold_vs_human"]
        models = evaluation["models_vs_human"]

    mv = find_row(pseudo, "Majority Pseudo-Gold")
    wv = find_row(pseudo, "Weighted Pseudo-Gold")
    lines += [f"[{label} MAIN TEST]"]
    if mv:
        lines.append(f"MV       P={fmt(mv['precision'])} R={fmt(mv['recall'])} F1={fmt(mv['f1'])}")
    if wv:
        lines.append(f"WV       P={fmt(wv['precision'])} R={fmt(wv['recall'])} F1={fmt(wv['f1'])}")
    if cawv:
        ca = cawv["overall"]["CAWV"]
        lines.append(f"CAWV     P={fmt(ca['precision'])} R={fmt(ca['recall'])} F1={fmt(ca['f1'])}")
        ci = cawv.get("bootstrap", {}).get("confidence_intervals_95", {}).get("CAWV")
        if ci:
            lines.append(f"CAWV 95% CI [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}]")
        for row in cawv.get("bootstrap", {}).get("comparisons", []):
            lines.append(
                f"{row['comparison']}: delta={float(row['f1_difference']):+.6f} "
                f"raw_p={fmt(row.get('raw_p'), 6)} Holm_p={fmt(row.get('holm_p'), 6)}"
            )
    for row in models:
        name = row.get("name", row.get("system"))
        lines.append(f"MODEL {name}: P={fmt(row['precision'])} R={fmt(row['recall'])} F1={fmt(row['f1'])}")
    lines.append("")


def add_overlap(lines: list[str]) -> None:
    root = ROOT / "results" / "biored_overlap_excluded"
    evaluation = load_json(root / "test" / "evaluation_results.json")
    cawv = load_json(root / "test" / "cawv" / "cawv_evaluation.json")
    audit = load_json(root / "overlap_audit.json")
    lines.append("[BIORED OVERLAP-EXCLUDED]")
    if audit:
        dev = audit["splits"]["dev"]
        test = audit["splits"]["test"]
        lines.append(
            f"Documents: DEV {dev['retained_documents']} retained / {dev['overlap_documents']} removed; "
            f"TEST {test['retained_documents']} retained / {test['overlap_documents']} removed"
        )
    if evaluation:
        pseudo = evaluation["pseudo_gold_vs_human"]
        mv = find_row(pseudo, "Majority Pseudo-Gold")
        wv = find_row(pseudo, "Weighted Pseudo-Gold")
        if mv:
            lines.append(f"MV       P={fmt(mv['precision'])} R={fmt(mv['recall'])} F1={fmt(mv['f1'])}")
        if wv:
            lines.append(f"WV       P={fmt(wv['precision'])} R={fmt(wv['recall'])} F1={fmt(wv['f1'])}")
    if cawv:
        ca = cawv["overall"]["CAWV"]
        lines.append(f"CAWV     P={fmt(ca['precision'])} R={fmt(ca['recall'])} F1={fmt(ca['f1'])}")
        ci = cawv.get("bootstrap", {}).get("confidence_intervals_95", {}).get("CAWV")
        if ci:
            lines.append(f"CAWV 95% CI [{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}]")
        for row in cawv.get("bootstrap", {}).get("comparisons", []):
            lines.append(
                f"{row['comparison']}: delta={float(row['f1_difference']):+.6f} "
                f"raw_p={fmt(row.get('raw_p'), 6)} Holm_p={fmt(row.get('holm_p'), 6)}"
            )
    else:
        lines.append("CAWV: NOT AVAILABLE")
    lines.append("")


def add_positive_token_diagnostics(lines: list[str]) -> None:
    lines.append("[POSITIVE-TOKEN DIAGNOSTICS — BC5CDR / BIORED / BIORED OVERLAP-EXCLUDED]")

    def emit(label: str, path: Path) -> None:
        payload = load_json(path)
        lines.append(label)
        if not payload:
            lines.append("NOT AVAILABLE")
            return

        models = payload.get("models")
        if isinstance(models, list):
            for row in models:
                lines.append(
                    f"{row['model']}: "
                    f"P={fmt(row.get('positive_token_precision'))} "
                    f"R={fmt(row.get('positive_token_recall'))} "
                    f"F1={fmt(row.get('positive_token_f1'))} "
                    f"kappa={fmt(row.get('bio_kappa'))}"
                )
            return

        if isinstance(models, dict):
            for model_name, model_payload in models.items():
                metrics = model_payload.get("positive_token_metrics", {})
                lines.append(
                    f"{model_name}: "
                    f"P={fmt(metrics.get('precision'))} "
                    f"R={fmt(metrics.get('recall'))} "
                    f"F1={fmt(metrics.get('f1'))} "
                    f"kappa={fmt(model_payload.get('multiclass_bio_kappa'))}"
                )
            return

        lines.append("NOT AVAILABLE")

    emit(
        "BC5CDR",
        ROOT / "results" / "test" / "kappa_diagnostic" / "kappa_diagnostic.json",
    )
    emit(
        "BIORED OFFICIAL",
        ROOT / "results" / "biored" / "test" / "kappa_diagnostic" / "kappa_diagnostic.json",
    )
    emit(
        "BIORED OVERLAP-EXCLUDED",
        ROOT
        / "results"
        / "biored_overlap_excluded"
        / "test"
        / "kappa_diagnostic"
        / "kappa_diagnostic.json",
    )
    lines.append("")

def add_ablation(lines: list[str], label: str, path: Path) -> None:
    payload = load_json(path)
    lines.append(f"[{label} CAWV ABLATION]")
    if payload is None:
        lines += ["NOT AVAILABLE", ""]
        return
    methods = payload.get("test_metrics") or payload.get("methods") or payload.get("test_methods") or payload.get("overall")
    if isinstance(methods, dict):
        for key, value in methods.items():
            if isinstance(value, dict) and "f1" in value:
                lines.append(f"{key}: F1={fmt(value['f1'])}")
    bootstrap = payload.get("bootstrap", {}) if isinstance(payload.get("bootstrap"), dict) else {}
    comparisons = bootstrap.get("comparisons", payload.get("comparisons", []))
    for row in comparisons:
        delta = row.get("f1_difference", row.get("delta_f1"))
        lo = row.get("difference_ci_low", row.get("delta_ci_low"))
        hi = row.get("difference_ci_high", row.get("delta_ci_high"))
        lines.append(
            f"{row.get('comparison')}: delta={fmt(delta, 6)} CI=[{fmt(lo, 6)}, {fmt(hi, 6)}] "
            f"raw_p={fmt(row.get('raw_p'), 6)} Holm_p={fmt(row.get('holm_p'), 6)}"
        )
    lines.append("")


def add_medgemma(lines: list[str], label: str, root: Path) -> None:
    evaluation = load_json(root / "test" / "medgemma_evaluation.json")
    cawv = load_json(root / "test" / "cawv_medgemma" / "cawv_medgemma_analysis.json")
    if cawv is None:
        candidates = list((root / "test").glob("**/*cawv*medgemma*analysis*.json")) if (root / "test").exists() else []
        if candidates:
            cawv = load_json(candidates[0])
    lines.append(f"[{label} MEDGEMMA]")
    if evaluation:
        for row in evaluation.get("results", []):
            lines.append(f"{row['system']}: P={fmt(row['precision'])} R={fmt(row['recall'])} F1={fmt(row['f1'])}")
    else:
        lines.append("WV/zero-shot MedGemma: NOT AVAILABLE")
    if cawv:
        # Keep this generic because the analysis schema may differ by branch/version.
        for key in ("baseline", "hybrid", "cawv", "cawv_medgemma"):
            value = cawv.get(key)
            if isinstance(value, dict) and "f1" in value:
                lines.append(f"{key}: F1={fmt(value['f1'])}")
        if "f1_difference" in cawv:
            lines.append(f"CAWV+MedGemma delta={float(cawv['f1_difference']):+.6f}")
    lines.append("")


def add_finetuned(lines: list[str]) -> None:
    lines.append("[FINE-TUNED MODELS — SYMMETRIC BC5CDR / BIORED]")
    combined = load_csv(ROOT / "results" / "finetuned_models" / "combined_performance.csv")
    boot = load_csv(ROOT / "results" / "finetuned_models" / "combined_cawv_vs_finetuned_bootstrap.csv")
    if not combined:
        lines += ["NOT AVAILABLE", ""]
        return
    for dataset in ("bc5cdr", "biored"):
        lines.append(dataset.upper())
        for row in combined:
            if row.get("dataset") != dataset:
                continue
            lines.append(
                f"{row['model']}: P={fmt(row['precision'])} R={fmt(row['recall'])} F1={fmt(row['f1'])} "
                f"95%CI=[{fmt(row.get('f1_ci_low'))}, {fmt(row.get('f1_ci_high'))}] "
                f"inference_s={fmt(row.get('inference_seconds'), 2)}"
            )
        public_cmp = load_json(ROOT / "results" / "finetuned_models" / dataset / "public_vs_finetuned_biobert.json")
        if public_cmp:
            lines.append(
                "BioBERT public vs fine-tuned: "
                f"public={fmt(public_cmp['public_f1'])} fine_tuned={fmt(public_cmp['finetuned_f1'])} "
                f"delta={float(public_cmp['delta_f1']):+.6f} "
                f"95%CI=[{fmt(public_cmp['delta_ci_low'], 6)}, {fmt(public_cmp['delta_ci_high'], 6)}] "
                f"p={fmt(public_cmp['raw_p'], 6)}"
            )
        lines.append("CAWV vs controlled fine-tuned bootstrap:")
        for row in boot:
            if row.get("dataset") != dataset:
                continue
            lines.append(
                f"  {row['comparison']}: delta={float(row['delta_f1']):+.6f} "
                f"95%CI=[{fmt(row['delta_ci_low'], 6)}, {fmt(row['delta_ci_high'], 6)}] "
                f"raw_p={fmt(row['raw_p'], 6)} Holm_p={fmt(row['holm_p'], 6)}"
            )
    lines.append("")


def add_resource_summary(lines: list[str]) -> None:
    lines.append("[RESOURCE / DEPLOYMENT MEASUREMENTS]")
    deployment = load_csv(ROOT / "results" / "resource_usage" / "deployment_resource_summary.csv")
    training = load_csv(ROOT / "results" / "resource_usage" / "finetuned_training_resource_summary.csv")
    cloud = load_csv(ROOT / "results" / "resource_usage" / "cloud_cost_inputs.csv")

    if deployment:
        lines.append("TEST / DEPLOYMENT")
        for row in deployment:
            status = row.get("measurement_status") or "UNKNOWN"
            runtime_seconds = row.get("runtime_seconds_for_cost")
            if runtime_seconds in (None, ""):
                lines.append(
                    f"{row.get('dataset')} | {row.get('method')}: "
                    f"{status}; runtime/cost basis NOT AVAILABLE"
                )
                continue
            lines.append(
                f"{row.get('dataset')} | {row.get('method')}: "
                f"status={status}; "
                f"runtime_basis_s={fmt(runtime_seconds, 2)} "
                f"s/doc={fmt(row.get('seconds_per_document'), 4)} "
                f"instance_hours={fmt(row.get('instance_hours_test'), 6)} "
                f"peak_RAM_MB={fmt(row.get('peak_process_tree_ram_mb_sequential'), 1)} "
                f"peak_GPU_VRAM_MB={fmt(row.get('peak_process_gpu_vram_mb_sequential'), 1)} "
                f"checkpoint_MB={fmt(row.get('checkpoint_storage_mb'), 1)} "
                f"checkpoints={row.get('underlying_checkpoint_count')}"
            )
            if row.get("runtime_basis"):
                lines.append(f"  basis: {row.get('runtime_basis')}")
            if row.get("resource_telemetry_status"):
                lines.append(f"  telemetry: {row.get('resource_telemetry_status')}")
    else:
        lines.append("Deployment resource summary: NOT AVAILABLE")

    if training:
        lines.append("FINE-TUNING")
        for row in training:
            lines.append(
                f"{row.get('dataset')} | {row.get('model')}: "
                f"two-head training_s={fmt(row.get('total_training_seconds_two_heads'), 2)} "
                f"GPU_hours={fmt(row.get('total_training_gpu_hours_two_heads'), 6)} "
                f"peak_RAM_MB={fmt(row.get('peak_training_ram_mb_sequential'), 1)} "
                f"peak_GPU_VRAM_MB={fmt(row.get('peak_training_gpu_vram_mb_sequential'), 1)} "
                f"checkpoint_MB={fmt(row.get('checkpoint_storage_mb_two_heads'), 1)}"
            )

    if cloud:
        lines.append(
            "APPROXIMATE CLOUD COMPUTE COST — representative public on-demand single-GPU rates "
            "checked 2026-10-13"
        )
        first = cloud[0]
        lines.append(
            f"Reference hourly range: "
            f"${float(first.get('reference_rate_low_usd_per_hour')):.3f}-"
            f"${float(first.get('reference_rate_high_usd_per_hour')):.3f}/hour; "
            f"middle reference ${float(first.get('reference_rate_mid_usd_per_hour')):.3f}/hour."
        )
        lines.append(
            "References: Google Cloud g2-standard-4/g2-standard-8 (NVIDIA L4) and "
            "AWS g5.xlarge (NVIDIA A10G). These are workload-capable references, not "
            "hardware-equivalent to the local RTX 3070 Ti."
        )
        lines.append(
            "Cost interpretation: observed local runtime is multiplied by the reference "
            "hourly rates. This is an illustrative compute-only estimate and does not "
            "model cloud-specific speed differences."
        )
        for row in cloud:
            if not str(row.get("estimated_test_cost_low_usd", "")).strip():
                continue
            lines.append(
                f"{row.get('dataset')} | {row.get('method')}: "
                f"approx_test_cost_USD="
                f"${float(row.get('estimated_test_cost_low_usd')):.4f}-"
                f"${float(row.get('estimated_test_cost_high_usd')):.4f}; "
                f"mid=${float(row.get('estimated_test_cost_mid_usd')):.4f}; "
                f"approx_per_1000_docs_USD="
                f"${float(row.get('estimated_cost_per_1000_docs_low_usd')):.4f}-"
                f"${float(row.get('estimated_cost_per_1000_docs_high_usd')):.4f}"
            )
        custom = [row for row in cloud if str(row.get("custom_estimated_test_instance_cost_usd", "")).strip()]
        if custom:
            lines.append("CUSTOM USER-SUPPLIED CLOUD RATE")
            for row in custom:
                lines.append(
                    f"{row.get('dataset')} | {row.get('method')}: "
                    f"custom_test_cost_USD={row.get('custom_estimated_test_instance_cost_usd')} "
                    f"custom_cost_per_1000_docs_USD={row.get('custom_estimated_instance_cost_per_1000_docs_usd')}"
                )

    lines.append(
        "CAVEAT: cached MedGemma replay wall time is excluded. Original saved request "
        "durations are used for MedGemma runtime/cost, while fresh MedGemma RAM/VRAM/energy "
        "remain unmeasured unless Ollama is deliberately rerun."
    )
    lines.append(
        "CAVEAT: measured resource values describe the actual sequential implementation. "
        "They do not establish universal lower total compute, memory, energy, or deployment cost."
    )
    lines.append("")


def generate() -> Path:
    lines = [
        "MANUSCRIPT RESULTS TRANSFER LOG",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "Use values only from sections marked available. Missing experiments are written explicitly as NOT AVAILABLE.",
        "",
    ]
    add_main_dataset(lines, "BC5CDR", ROOT / "results")
    add_main_dataset(lines, "BIORED OFFICIAL", ROOT / "results" / "biored")
    add_overlap(lines)
    add_positive_token_diagnostics(lines)
    add_ablation(lines, "BC5CDR", ROOT / "results" / "test" / "cawv" / "ablation" / "cawv_ablation_results.json")
    add_ablation(lines, "BIORED OFFICIAL", ROOT / "results" / "biored" / "test" / "cawv" / "ablation" / "cawv_ablation_results.json")
    add_ablation(lines, "BIORED OVERLAP-EXCLUDED", ROOT / "results" / "biored_overlap_excluded" / "test" / "cawv" / "ablation" / "cawv_ablation_results.json")
    add_finetuned(lines)
    add_medgemma(lines, "BC5CDR", ROOT / "results")
    add_medgemma(lines, "BIORED OFFICIAL", ROOT / "results" / "biored")
    add_medgemma(lines, "BIORED OVERLAP-EXCLUDED", ROOT / "results" / "biored_overlap_excluded")
    add_resource_summary(lines)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Manuscript transfer log: {OUTPUT}")
    return OUTPUT


def main() -> None:
    generate()


if __name__ == "__main__":
    main()
