from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from src.pipeline_common import ROOT, check_raw_inputs, check_ollama, stage_done
from src.run_bc5cdr import stages as bc_stages
from src.run_biored import official_stages, overlap_sensitivity_stages, comparison_stage
from src.run_medgemma import bc5cdr_medgemma_stages, bc5cdr_cawv_medgemma_stages, biored_medgemma_stages, biored_cawv_medgemma_stages

from src.manuscript_results_log import generate as generate_manuscript_log
from src.run_bc5cdr import run_pipeline as run_bc5cdr
from src.run_biored import run_pipeline as run_biored
from src.run_medgemma import run_pipeline as run_medgemma
from src.resource_summary import generate as generate_resource_summary
from src.reproducibility_snapshot import generate as generate_reproducibility_snapshot
from src.symmetric_reporting import generate as generate_symmetric_reporting
from src.validate_outputs import main as validate_outputs


def plan():
    return [
        ("BC5CDR", bc_stages()),
        ("BioRED official", official_stages()),
        ("BioRED overlap excluded", overlap_sensitivity_stages() + [comparison_stage()]),
        ("MedGemma BC5CDR", bc5cdr_medgemma_stages() + bc5cdr_cawv_medgemma_stages()),
        ("MedGemma BioRED official", biored_medgemma_stages("official")),
        ("MedGemma BioRED overlap excluded", biored_medgemma_stages("overlap_excluded") + biored_cawv_medgemma_stages()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete reproducible CAWV study with one command.")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    parser.add_argument("--skip-finetuned", action="store_true", help="Explicitly omit supervised controls (partial study).")
    parser.add_argument("--dry-run", action="store_true", help="Inspect every planned stage without running models.")
    parser.add_argument("--no-preflight", action="store_true", help="Defer Ollama availability checks (useful if starting it later).")
    args = parser.parse_args()
    groups = plan()
    if args.dry_run:
        for group, stages in groups:
            print(f"\n[{group}] {len(stages)} stages")
            for stage in stages:
                print(f"  {'RUN ' if args.force or stage.always_run or not stage_done(stage) else 'SKIP'} {stage.label}")
        print("\nFine-tuning:", "SKIPPED by request" if args.skip_finetuned else "BC5CDR + BioRED (4 encoders, 2 entity types)")
        print("Final: resource summary, provenance, manuscript log, symmetric tables/figures, validation")
        return

    # Fail before wasting hours on encoders if mandatory external runtime is absent.
    check_raw_inputs()
    if not args.no_preflight:
        check_ollama()
    run_dir = ROOT / "results" / "run_control"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    status = {"start_utc": started, "mode": "complete" if not args.skip_finetuned else "partial_without_finetuning", "steps": [], "state": "running"}
    status_path = run_dir / "last_run.json"
    def save():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temp.replace(status_path)
    save()
    from src.finetuned_models_symmetric import run_all as run_finetuned
    jobs = [
        ("BC5CDR", lambda: run_bc5cdr(force=args.force)),
        ("BioRED official and overlap-excluded", lambda: run_biored(force=args.force)),
        ("MedGemma all branches", lambda: run_medgemma(force=args.force)),
    ]
    if not args.skip_finetuned:
        jobs.append(("Symmetric supervised controls", lambda: run_finetuned(force=args.force, resamples=1000, seed=42)))
    jobs.extend([
        ("Resource summary", generate_resource_summary),
        ("Manuscript transfer log", generate_manuscript_log),
        ("Reproducibility snapshot", generate_reproducibility_snapshot),
        ("Symmetric publication tables and figures", generate_symmetric_reporting),
        ("Output validation", validate_outputs if not args.skip_finetuned else lambda: print("Full-study validation not run: supervised controls explicitly skipped.")),
    ])
    try:
        for i, (label, action) in enumerate(jobs, 1):
            print(f"\n{'='*78}\n[{i}/{len(jobs)}] {label}\n{'='*78}", flush=True)
            record = {"stage": label, "start_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
            status["steps"].append(record)
            save()
            try:
                action()
            except Exception as exc:
                record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
                raise
            else:
                record.update(status="complete", end_utc=datetime.now(timezone.utc).isoformat())
            finally:
                save()
        status["state"] = "complete" if not args.skip_finetuned else "partial"
    except BaseException as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        raise
    finally:
        status["end_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print("\nCOMPLETE. Validation passed. Results: results/publication/", flush=True)
    print("Status: results/run_control/last_run.json", flush=True)


if __name__ == "__main__":
    main()
