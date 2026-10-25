from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import urllib.request
from pathlib import Path

from src.experiment_config import PROJECT_ROOT as ROOT
from src.resource_monitor import RESOURCE_DIR, latest_stage_records

DEPLOYMENT_CSV = RESOURCE_DIR / "deployment_resource_summary.csv"
TRAINING_CSV = RESOURCE_DIR / "finetuned_training_resource_summary.csv"
CLOUD_CSV = RESOURCE_DIR / "cloud_cost_inputs.csv"
SYSTEM_JSON = RESOURCE_DIR / "system_info.json"
NOTES_MD = RESOURCE_DIR / "RESOURCE_MEASUREMENT_NOTES.md"


# Representative public on-demand single-GPU VM rates checked 2026-10-13.
# These are cost references, not claims of hardware equivalence to the local RTX 3070 Ti.
# GCP source: official accelerator-optimized VM pricing page.
# AWS source: AWS Marketplace listing for a standard GPU deep-learning image; the listing
# reports the underlying g5.xlarge on-demand compute rate as approximately $1.006/hour.
REFERENCE_CLOUD_RATES = [
    {
        "provider": "Google Cloud",
        "instance": "g2-standard-4",
        "gpu": "NVIDIA L4 24 GB",
        "hourly_usd": 0.706832276,
        "pricing_basis": "public on-demand VM rate",
        "source": "https://cloud.google.com/products/compute/pricing/accelerator-optimized",
        "checked_date": "2026-10-13",
    },
    {
        "provider": "Google Cloud",
        "instance": "g2-standard-8",
        "gpu": "NVIDIA L4 24 GB",
        "hourly_usd": 0.853624312,
        "pricing_basis": "public on-demand VM rate",
        "source": "https://cloud.google.com/products/compute/pricing/accelerator-optimized",
        "checked_date": "2026-10-13",
    },
    {
        "provider": "Amazon Web Services",
        "instance": "g5.xlarge",
        "gpu": "NVIDIA A10G 24 GB",
        "hourly_usd": 1.006,
        "pricing_basis": "approximate public on-demand rate",
        "source": "https://aws.amazon.com/marketplace/pp/prodview-g3h4ayexm4svi",
        "checked_date": "2026-10-13",
    },
]

REFERENCE_MIN_RATE = min(item["hourly_usd"] for item in REFERENCE_CLOUD_RATES)
REFERENCE_MAX_RATE = max(item["hourly_usd"] for item in REFERENCE_CLOUD_RATES)
REFERENCE_MID_RATE = sorted(item["hourly_usd"] for item in REFERENCE_CLOUD_RATES)[1]


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nvidia_info():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return []
        rows = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 3:
                rows.append(
                    {
                        "name": parts[0],
                        "memory_total_mb": float(parts[1]),
                        "driver_version": parts[2],
                    }
                )
        return rows
    except Exception:
        return []


def _ollama_medgemma_size():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            payload = json.load(response)
        for item in payload.get("models", []):
            name = str(item.get("name") or item.get("model") or "")
            if name == "medgemma1.5:4b-it-q4_K_M":
                return {
                    "name": name,
                    "size_bytes": item.get("size"),
                    "size_gb": (
                        float(item["size"]) / (1024 ** 3)
                        if item.get("size") is not None
                        else None
                    ),
                }
    except Exception:
        return None
    return None


def system_info():
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
    except Exception:
        ram_gb = None
        physical_cores = None
        logical_cores = os.cpu_count()
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "physical_cpu_cores": physical_cores,
        "logical_cpu_cores": logical_cores,
        "system_ram_gb": ram_gb,
        "gpus": _nvidia_info(),
        "ollama_medgemma": _ollama_medgemma_size(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info


def _max(rows, field):
    values = [float(row[field]) for row in rows if row and row.get(field) is not None]
    return max(values) if values else None


def _sum(rows, field):
    values = [float(row[field]) for row in rows if row and row.get(field) is not None]
    return sum(values) if values else None


def _deployment_row(
    *,
    dataset: str,
    method: str,
    documents: int,
    stage_rows: list[dict],
    pipelines,
    checkpoints,
    checkpoint_storage_mb=None,
    notes="",
):
    successful_rows = []
    for row in stage_rows:
        if not row:
            continue
        try:
            if int(row.get("return_code", 0)) != 0:
                continue
        except Exception:
            continue
        successful_rows.append(row)

    if len(successful_rows) < int(pipelines):
        return {
            "dataset": dataset,
            "method": method,
            "test_documents": documents,
            "pipeline_count": pipelines,
            "underlying_checkpoint_count": checkpoints,
            "measurement_status": "INCOMPLETE RESOURCE MEASUREMENT",
            "runtime_basis": (
                f"Only {len(successful_rows)} of {int(pipelines)} required successful "
                "pipeline resource records are available."
            ),
            "resource_telemetry_status": "INCOMPLETE",
            "checkpoint_storage_mb": checkpoint_storage_mb,
            "notes": notes,
        }

    stage_rows = successful_rows
    wall = _sum(stage_rows, "wall_seconds")
    if wall is None:
        return {
            "dataset": dataset,
            "method": method,
            "test_documents": documents,
            "pipeline_count": pipelines,
            "underlying_checkpoint_count": checkpoints,
            "measurement_status": "NOT MEASURED IN THIS RUN",
            "notes": notes,
        }
    return {
        "dataset": dataset,
        "method": method,
        "test_documents": documents,
        "pipeline_count": pipelines,
        "underlying_checkpoint_count": checkpoints,
        "measurement_status": "MEASURED",
        "runtime_basis": "Fresh monitored stage wall time.",
        "resource_telemetry_status": "Fresh stage RAM/VRAM/energy telemetry when available.",
        "end_to_end_test_seconds": wall,
        "runtime_seconds_for_cost": wall,
        "seconds_per_document": wall / max(1, documents),
        "documents_per_second": documents / wall if wall > 0 else None,
        "instance_hours_test": wall / 3600.0,
        "instance_hours_per_1000_documents": (wall / max(1, documents)) * 1000.0 / 3600.0,
        "peak_process_tree_ram_mb_sequential": _max(stage_rows, "peak_process_tree_ram_mb"),
        "peak_process_gpu_vram_mb_sequential": _max(stage_rows, "peak_process_gpu_vram_mb"),
        "peak_total_gpu_memory_used_mb": _max(stage_rows, "peak_total_gpu_memory_used_mb"),
        "approx_system_gpu_energy_wh": _sum(stage_rows, "approx_system_gpu_energy_wh"),
        "checkpoint_storage_mb": checkpoint_storage_mb,
        "cached_replay_telemetry_excluded": False,
        "notes": notes,
    }


def _runtime_fields(seconds: float, documents: int) -> dict:
    seconds = float(seconds)
    return {
        "runtime_seconds_for_cost": seconds,
        "seconds_per_document": seconds / max(1, documents),
        "documents_per_second": documents / seconds if seconds > 0 else None,
        "instance_hours_test": seconds / 3600.0,
        "instance_hours_per_1000_documents": (
            (seconds / max(1, documents)) * 1000.0 / 3600.0
        ),
    }


def _historical_medgemma_zero_shot_row(
    *,
    dataset: str,
    documents: int,
    summary_path: Path,
    checkpoint_storage_mb,
) -> dict:
    summary = load_json(summary_path)
    seconds = None if not summary else summary.get("summed_request_seconds")
    if seconds is None:
        return {
            "dataset": dataset,
            "method": "MedGemma zero-shot",
            "test_documents": documents,
            "pipeline_count": 1,
            "underlying_checkpoint_count": 1,
            "measurement_status": "NOT AVAILABLE",
            "runtime_basis": "No saved original MedGemma request-duration summary was found.",
            "resource_telemetry_status": "NOT AVAILABLE",
            "checkpoint_storage_mb": checkpoint_storage_mb,
        }

    row = {
        "dataset": dataset,
        "method": "MedGemma zero-shot",
        "test_documents": documents,
        "pipeline_count": 1,
        "underlying_checkpoint_count": 1,
        "measurement_status": "HISTORICAL REQUEST TIME",
        "runtime_basis": (
            "Summed Ollama request durations saved during the original MedGemma inference. "
            "A later cached replay is not used as inference runtime."
        ),
        "resource_telemetry_status": (
            "RAM/VRAM/energy NOT measured during fresh MedGemma inference; "
            "cached replay telemetry is intentionally excluded."
        ),
        "end_to_end_test_seconds": None,
        "peak_process_tree_ram_mb_sequential": None,
        "peak_process_gpu_vram_mb_sequential": None,
        "peak_total_gpu_memory_used_mb": None,
        "approx_system_gpu_energy_wh": None,
        "checkpoint_storage_mb": checkpoint_storage_mb,
        "cached_replay_telemetry_excluded": True,
        "historical_request_seconds": float(seconds),
        "notes": (
            "Runtime/cost basis is the original saved Ollama request duration. "
            "It excludes cached replay time and does not provide fresh memory or energy telemetry."
        ),
    }
    row.update(_runtime_fields(float(seconds), documents))
    return row


def _historical_hybrid_request_seconds(summary_path: Path, field: str):
    summary = load_json(summary_path)
    if not summary:
        return None
    value = summary.get(field)
    return float(value) if value is not None else None


def _mixed_hybrid_row(
    *,
    dataset: str,
    method: str,
    documents: int,
    conventional_rows: list[dict],
    llm_summary_path: Path,
    llm_seconds_field: str,
    checkpoint_storage_mb,
) -> dict:
    conventional = []
    for row in conventional_rows:
        if not row:
            continue
        try:
            if int(row.get("return_code", 0)) != 0:
                continue
        except Exception:
            continue
        conventional.append(row)

    if len(conventional) < 5:
        return {
            "dataset": dataset,
            "method": method,
            "test_documents": documents,
            "pipeline_count": 6,
            "underlying_checkpoint_count": 8,
            "measurement_status": "INCOMPLETE RESOURCE MEASUREMENT",
            "runtime_basis": (
                f"Only {len(conventional)} of 5 required conventional pipeline "
                "resource records are successful."
            ),
            "resource_telemetry_status": "INCOMPLETE",
            "checkpoint_storage_mb": checkpoint_storage_mb,
        }

    conventional_seconds = _sum(conventional, "wall_seconds")
    llm_seconds = _historical_hybrid_request_seconds(
        llm_summary_path, llm_seconds_field
    )
    if conventional_seconds is None or llm_seconds is None:
        return {
            "dataset": dataset,
            "method": method,
            "test_documents": documents,
            "pipeline_count": 6,
            "underlying_checkpoint_count": 8,
            "measurement_status": "PARTIAL / NOT AVAILABLE",
            "runtime_basis": (
                "Needs both monitored conventional inference and saved original "
                "MedGemma tie-break request durations."
            ),
            "resource_telemetry_status": "INCOMPLETE",
            "checkpoint_storage_mb": checkpoint_storage_mb,
        }

    total_seconds = float(conventional_seconds) + float(llm_seconds)
    row = {
        "dataset": dataset,
        "method": method,
        "test_documents": documents,
        "pipeline_count": 6,
        "underlying_checkpoint_count": 8,
        "measurement_status": "MIXED MEASUREMENT",
        "runtime_basis": (
            "Fresh monitored conventional-pipeline wall time plus original saved "
            "MedGemma tie-break request durations. Cached replay wall time is excluded."
        ),
        "resource_telemetry_status": (
            "Conventional RAM/VRAM measured fresh. Complete hybrid peak memory/energy "
            "is NOT reported because MedGemma was not rerun fresh."
        ),
        "end_to_end_test_seconds": None,
        "conventional_inference_seconds": float(conventional_seconds),
        "historical_medgemma_request_seconds": float(llm_seconds),
        "peak_process_tree_ram_mb_sequential": None,
        "peak_process_gpu_vram_mb_sequential": None,
        "peak_total_gpu_memory_used_mb": None,
        "approx_system_gpu_energy_wh": None,
        "checkpoint_storage_mb": checkpoint_storage_mb,
        "cached_replay_telemetry_excluded": True,
        "notes": (
            "Cost/runtime total combines observed conventional inference with the "
            "original LLM request durations. Full hybrid RAM/VRAM/energy remains unmeasured."
        ),
    }
    row.update(_runtime_fields(total_seconds, documents))
    return row


def _stage(label, records):
    row = records.get(label)
    if not row:
        return None
    try:
        if int(row.get("return_code", 0)) != 0:
            return None
    except Exception:
        return None
    return row


def deployment_rows():
    records = latest_stage_records()
    rows = []

    bc5_model_labels = [
        f"BC5CDR test model: {key}"
        for key in ("scispacy", "biobert", "pubmedbert", "clinicalbert", "d4data")
    ]
    br_model_labels = [
        f"BioRED test model: {key}"
        for key in ("scispacy", "biobert", "pubmedbert", "clinicalbert", "d4data")
    ]
    bc5_models = [_stage(label, records) for label in bc5_model_labels]
    br_models = [_stage(label, records) for label in br_model_labels]

    rows.append(
        _deployment_row(
            dataset="BC5CDR",
            method="WV/CAWV conventional ensemble input generation",
            documents=500,
            stage_rows=[row for row in bc5_models if row],
            pipelines=5,
            checkpoints=7,
            notes="Five released pipelines. BioBERT and PubMedBERT+OpenMed each contain two target checkpoints. Stages were measured sequentially.",
        )
    )
    rows.append(
        _deployment_row(
            dataset="BioRED official",
            method="WV/CAWV conventional ensemble input generation",
            documents=100,
            stage_rows=[row for row in br_models if row],
            pipelines=5,
            checkpoints=7,
            notes="Five released pipelines. BioBERT and PubMedBERT+OpenMed each contain two target checkpoints. Stages were measured sequentially.",
        )
    )

    med_size = _ollama_medgemma_size()
    med_storage = (
        med_size.get("size_gb") * 1024
        if med_size and med_size.get("size_gb") is not None
        else None
    )

    rows.append(
        _historical_medgemma_zero_shot_row(
            dataset="BC5CDR",
            documents=500,
            summary_path=ROOT / "results" / "test" / "medgemma" / "zero_shot_summary.json",
            checkpoint_storage_mb=med_storage,
        )
    )
    rows.append(
        _historical_medgemma_zero_shot_row(
            dataset="BioRED official",
            documents=100,
            summary_path=ROOT / "results" / "biored" / "test" / "medgemma" / "zero_shot_summary.json",
            checkpoint_storage_mb=med_storage,
        )
    )
    rows.append(
        _historical_medgemma_zero_shot_row(
            dataset="BioRED overlap-excluded",
            documents=66,
            summary_path=ROOT / "results" / "biored_overlap_excluded" / "test" / "medgemma" / "zero_shot_summary.json",
            checkpoint_storage_mb=med_storage,
        )
    )

    # Selective hybrid totals use fresh conventional inference telemetry plus the
    # original saved LLM request durations. Cached replay wall time is never treated
    # as MedGemma inference time.
    rows.append(
        _mixed_hybrid_row(
            dataset="BC5CDR",
            method="WV + selective MedGemma",
            documents=500,
            conventional_rows=bc5_models,
            llm_summary_path=ROOT / "results" / "test" / "medgemma_hybrid" / "hybrid_summary.json",
            llm_seconds_field="tie_break_request_seconds",
            checkpoint_storage_mb=med_storage,
        )
    )
    rows.append(
        _mixed_hybrid_row(
            dataset="BC5CDR",
            method="CAWV + selective MedGemma",
            documents=500,
            conventional_rows=bc5_models,
            llm_summary_path=ROOT / "results" / "test" / "medgemma_cawv_hybrid" / "hybrid_summary.json",
            llm_seconds_field="routed_seconds",
            checkpoint_storage_mb=med_storage,
        )
    )
    rows.append(
        _mixed_hybrid_row(
            dataset="BioRED official",
            method="WV + selective MedGemma",
            documents=100,
            conventional_rows=br_models,
            llm_summary_path=ROOT / "results" / "biored" / "test" / "medgemma_hybrid" / "hybrid_summary.json",
            llm_seconds_field="tie_break_request_seconds",
            checkpoint_storage_mb=med_storage,
        )
    )
    rows.append(
        _mixed_hybrid_row(
            dataset="BioRED official",
            method="CAWV + selective MedGemma",
            documents=100,
            conventional_rows=br_models,
            llm_summary_path=ROOT / "results" / "biored" / "test" / "medgemma_cawv_hybrid" / "hybrid_summary.json",
            llm_seconds_field="routed_seconds",
            checkpoint_storage_mb=med_storage,
        )
    )
    rows.append(
        _mixed_hybrid_row(
            dataset="BioRED overlap-excluded",
            method="WV + selective MedGemma",
            documents=66,
            conventional_rows=[],
            llm_summary_path=ROOT / "results" / "biored_overlap_excluded" / "test" / "medgemma_hybrid" / "hybrid_summary.json",
            llm_seconds_field="tie_break_request_seconds",
            checkpoint_storage_mb=med_storage,
        )
    )

    # Fine-tuned two-checkpoint deployment rows.
    combined_path = ROOT / "results" / "finetuned_models" / "combined_performance.csv"
    if combined_path.exists():
        with combined_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                dataset_key = item.get("dataset")
                docs = 500 if dataset_key == "bc5cdr" else 100
                dataset = "BC5CDR" if dataset_key == "bc5cdr" else "BioRED official"
                wall = item.get("end_to_end_inference_seconds")
                if wall in (None, ""):
                    continue
                wall = float(wall)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": item["model"],
                        "test_documents": docs,
                        "pipeline_count": 1,
                        "underlying_checkpoint_count": int(float(item.get("checkpoint_count") or 2)),
                        "measurement_status": "MEASURED",
                        "runtime_basis": "Fresh monitored two-checkpoint sequential test inference.",
                        "resource_telemetry_status": "Fresh RAM/VRAM/energy telemetry when available.",
                        "end_to_end_test_seconds": wall,
                        "runtime_seconds_for_cost": wall,
                        "seconds_per_document": wall / docs,
                        "documents_per_second": docs / wall if wall > 0 else None,
                        "instance_hours_test": wall / 3600.0,
                        "instance_hours_per_1000_documents": (wall / docs) * 1000.0 / 3600.0,
                        "peak_process_tree_ram_mb_sequential": item.get("peak_inference_ram_mb"),
                        "peak_process_gpu_vram_mb_sequential": item.get("peak_inference_gpu_vram_mb"),
                        "peak_total_gpu_memory_used_mb": item.get("peak_total_gpu_memory_used_mb"),
                        "approx_system_gpu_energy_wh": item.get("approx_system_gpu_energy_wh"),
                        "checkpoint_storage_mb": item.get("checkpoint_storage_mb"),
                        "cached_replay_telemetry_excluded": False,
                        "notes": "One encoder family deployed as two independently fine-tuned checkpoints: DISEASE and CHEMICAL. The two heads are run sequentially in this implementation.",
                    }
                )
    return rows


def training_rows():
    rows = []
    root = ROOT / "results" / "finetuned_models"
    for dataset in ("bc5cdr", "biored"):
        for model_key in ("biobert", "biomedbert", "bert_base", "distilbert"):
            targets = []
            for target in ("disease", "chemical"):
                payload = load_json(root / dataset / model_key / target / "training_complete.json")
                if payload:
                    targets.append(payload)
            if not targets:
                continue
            rows.append(
                {
                    "dataset": "BC5CDR" if dataset == "bc5cdr" else "BioRED official",
                    "model": targets[0].get("model"),
                    "target_checkpoints": len(targets),
                    "total_training_seconds_two_heads": sum(float(x.get("training_seconds") or 0.0) for x in targets),
                    "total_training_gpu_hours_two_heads": sum(float(x.get("training_gpu_hours") or 0.0) for x in targets),
                    "peak_training_ram_mb_sequential": max(
                        [float(x["training_peak_ram_mb"]) for x in targets if x.get("training_peak_ram_mb") is not None],
                        default=None,
                    ),
                    "peak_training_gpu_vram_mb_sequential": max(
                        [float(x["training_peak_gpu_vram_mb"]) for x in targets if x.get("training_peak_gpu_vram_mb") is not None],
                        default=None,
                    ),
                    "peak_total_gpu_memory_used_mb": max(
                        [float(x["training_peak_total_gpu_memory_used_mb"]) for x in targets if x.get("training_peak_total_gpu_memory_used_mb") is not None],
                        default=None,
                    ),
                    "approx_system_gpu_energy_wh": sum(
                        float(x.get("training_approx_system_gpu_energy_wh") or 0.0) for x in targets
                    ) or None,
                    "checkpoint_storage_mb_two_heads": sum(float(x.get("checkpoint_storage_mb") or 0.0) for x in targets),
                    "parameter_count_per_head": max(int(x.get("parameter_count") or 0) for x in targets),
                    "notes": "Training resources are summed across the independently trained DISEASE and CHEMICAL checkpoints; peak memory is the maximum sequential peak.",
                }
            )
    return rows


def cloud_rows(deployment: list[dict]):
    rate_text = os.environ.get("CLOUD_INSTANCE_HOURLY_USD", "").strip()
    try:
        custom_rate = float(rate_text) if rate_text else None
    except Exception:
        custom_rate = None

    rows = []
    for item in deployment:
        hours = item.get("instance_hours_test")
        h1000 = item.get("instance_hours_per_1000_documents")
        runtime_basis = item.get("runtime_basis")

        row = dict(item)
        row["reference_rate_low_usd_per_hour"] = REFERENCE_MIN_RATE
        row["reference_rate_mid_usd_per_hour"] = REFERENCE_MID_RATE
        row["reference_rate_high_usd_per_hour"] = REFERENCE_MAX_RATE

        if hours is not None:
            hours = float(hours)
            row["estimated_test_cost_low_usd"] = hours * REFERENCE_MIN_RATE
            row["estimated_test_cost_mid_usd"] = hours * REFERENCE_MID_RATE
            row["estimated_test_cost_high_usd"] = hours * REFERENCE_MAX_RATE
        else:
            row["estimated_test_cost_low_usd"] = None
            row["estimated_test_cost_mid_usd"] = None
            row["estimated_test_cost_high_usd"] = None

        if h1000 is not None:
            h1000 = float(h1000)
            row["estimated_cost_per_1000_docs_low_usd"] = h1000 * REFERENCE_MIN_RATE
            row["estimated_cost_per_1000_docs_mid_usd"] = h1000 * REFERENCE_MID_RATE
            row["estimated_cost_per_1000_docs_high_usd"] = h1000 * REFERENCE_MAX_RATE
        else:
            row["estimated_cost_per_1000_docs_low_usd"] = None
            row["estimated_cost_per_1000_docs_mid_usd"] = None
            row["estimated_cost_per_1000_docs_high_usd"] = None

        row["cost_runtime_basis"] = runtime_basis
        row["cost_estimate_interpretation"] = (
            "Illustrative compute-only estimate obtained by multiplying the observed local "
            "runtime basis by representative cloud hourly prices. This does not model "
            "cloud-specific speed differences."
        )
        row["reference_cloud_rate_statement"] = (
            f"Representative public on-demand single-GPU VM range checked 2026-10-13: "
            f"${REFERENCE_MIN_RATE:.3f}-${REFERENCE_MAX_RATE:.3f}/hour; "
            f"mid reference ${REFERENCE_MID_RATE:.3f}/hour."
        )
        row["reference_rate_scope"] = (
            "Comparable single-GPU cloud instances capable of the workload; "
            "not hardware-equivalent to the local RTX 3070 Ti."
        )

        row["custom_cloud_instance_hourly_rate_usd"] = custom_rate
        row["custom_estimated_test_instance_cost_usd"] = (
            float(hours) * custom_rate if custom_rate is not None and hours is not None else None
        )
        row["custom_estimated_instance_cost_per_1000_docs_usd"] = (
            float(h1000) * custom_rate if custom_rate is not None and h1000 is not None else None
        )
        row["cloud_cost_scope"] = (
            "Compute-time estimate only. Excludes storage, network/egress, startup/idle time, "
            "taxes, discounts, spot pricing, reservations, and provider-specific billing rules."
        )
        rows.append(row)
    return rows


def write_reference_cloud_rates():
    path = RESOURCE_DIR / "reference_cloud_gpu_rates.csv"
    write_csv(path, REFERENCE_CLOUD_RATES)
    return path


def generate():
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_JSON.write_text(json.dumps(system_info(), indent=2) + "\n", encoding="utf-8")
    deployment = deployment_rows()
    training = training_rows()
    cloud = cloud_rows(deployment)
    reference_rate_path = write_reference_cloud_rates()
    write_csv(DEPLOYMENT_CSV, deployment)
    write_csv(TRAINING_CSV, training)
    write_csv(CLOUD_CSV, cloud)

    NOTES_MD.write_text(
        """# Resource measurement notes

The repository distinguishes measured resource data from claims that remain unsupported.

## Measured when available

- End-to-end wall time for monitored stages.
- Test throughput and instance/GPU-hours derived from measured wall time.
- Peak process-tree RAM using RSS.
- Peak process GPU VRAM using NVIDIA process accounting.
- Peak total GPU memory used during the stage.
- Approximate whole-GPU energy from `nvidia-smi` power telemetry.
- Fine-tuned checkpoint storage size.
- Parameter count for each fine-tuned head.
- Fine-tuned training time and test inference time.

The supervised reference for each encoder uses **two independently fine-tuned checkpoints**:
one DISEASE model and one CHEMICAL model. Training time and checkpoint storage are summed
across the two heads. Peak RAM/VRAM is the maximum sequential peak because the current
implementation loads the two heads one at a time.

## Cached MedGemma handling

MedGemma zero-shot and selective-routing outputs may already exist in the Ollama decision
cache. A cached replay can finish in seconds even though the original LLM inference took
minutes or hours. **Cached replay wall time is never used as MedGemma inference cost.**

For MedGemma, runtime and cost calculations use the original per-request durations stored
inside the zero-shot or tie-break cache summaries. If MedGemma was not rerun fresh, RAM,
VRAM, total GPU-memory peak, and energy are reported as not measured rather than copied
from the cached replay.

For hybrid methods, the runtime/cost basis combines freshly monitored conventional-pipeline
wall time with the original saved MedGemma request durations. Full hybrid RAM/VRAM/energy
is left unreported unless fresh LLM inference is performed.

## Important limits

The GPU energy value is system-level telemetry and is not process-isolated. It should be
treated as an approximate diagnostic, not a precision energy benchmark.

Peak memory is measured for the actual sequential implementation. It is not an estimate of
memory required if all models were loaded concurrently.

The repository does not claim FLOPs, carbon emissions, or universal deployment cost.

## Cloud-cost calculation

`cloud_cost_inputs.csv` always contains the measured instance-hours needed for the observed
test run and the normalised instance-hours per 1,000 documents.

The repository includes a dated representative public on-demand single-GPU rate range
checked on 2026-10-13. The references are Google Cloud g2-standard-4 and g2-standard-8
with NVIDIA L4 GPUs and AWS g5.xlarge with an NVIDIA A10G GPU. These are practical
single-GPU cloud references capable of the workload, not hardware-equivalent replacements
for the local RTX 3070 Ti.

The default report therefore gives a low, middle, and high approximate compute-only cost
by multiplying measured instance-hours by the reference rates. The reference prices are
stored in `reference_cloud_gpu_rates.csv` with provider, instance, GPU, date, and source.

An optional current provider-specific rate can still be supplied:

```cmd
set CLOUD_INSTANCE_HOURLY_USD=YOUR_CURRENT_INSTANCE_PRICE
RUN_ALL.cmd
```

Cloud estimates exclude storage, data transfer, idle/startup time, taxes, discounts, spot
pricing, reservations, and provider-specific billing rules.
""",
        encoding="utf-8",
    )
    print(f"Resource summary: {DEPLOYMENT_CSV}")
    print(f"Fine-tuning resource summary: {TRAINING_CSV}")
    print(f"Cloud-cost inputs: {CLOUD_CSV}")
    print(f"Reference cloud GPU rates: {reference_rate_path}")
    print(
        "Representative on-demand single-GPU reference range: "
        f"${REFERENCE_MIN_RATE:.3f}-${REFERENCE_MAX_RATE:.3f}/hour "
        f"(checked 2026-10-13)."
    )
    return DEPLOYMENT_CSV


def main():
    generate()


if __name__ == "__main__":
    main()
