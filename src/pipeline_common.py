
from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_KEYS = ("scispacy", "biobert", "pubmedbert", "clinicalbert", "d4data")
MEDGEMMA_MODEL = "medgemma1.5:4b-it-q4_K_M"


@dataclass(frozen=True)
class Stage:
    label: str
    module: str
    args: tuple[str, ...]
    outputs: tuple[Path, ...]
    env: dict[str, str] | None = None
    always_run: bool = False
    resource_required: bool = False


def biored_dirs(variant: str) -> tuple[Path, Path, Path]:
    if variant == "official":
        return (
            ROOT / "data" / "processed" / "biored",
            ROOT / "data" / "gold" / "biored",
            ROOT / "results" / "biored",
        )
    if variant == "overlap_excluded":
        return (
            ROOT / "data" / "processed" / "biored_overlap_excluded",
            ROOT / "data" / "gold" / "biored_overlap_excluded",
            ROOT / "results" / "biored_overlap_excluded",
        )
    raise ValueError(f"Unknown BioRED variant: {variant}")


def biored_env(variant: str) -> dict[str, str]:
    return {"BIORED_VARIANT": variant}


def stage_done(stage: Stage) -> bool:
    """Never reuse empty, partial, or older-than-source stage products."""
    if not stage.outputs or any(not path.is_file() or path.stat().st_size == 0 for path in stage.outputs):
        return False
    # A modified stage script invalidates its old generated outputs.
    spec = importlib.util.find_spec(stage.module)
    if spec and spec.origin:
        source = Path(spec.origin)
        if source.exists() and min(p.stat().st_mtime_ns for p in stage.outputs) < source.stat().st_mtime_ns:
            return False
    if stage.module in {"src.biored_models", "src.scispacy_bc5cdr", "src.biobert_bc5cdr", "src.pubmed_bc5cdr", "src.clinicalbert_bc5cdr", "src.d4data_bc5cdr"}:
        loader_sources = [Path(__file__).parent / "cpu_ner.py", Path(__file__).parent / "ner_pipeline_labels.py"]
        if any(p.exists() and min(out.stat().st_mtime_ns for out in stage.outputs) < p.stat().st_mtime_ns for p in loader_sources):
            return False
    # Runtime policy changes must invalidate older cached stage products.
    common_source = Path(__file__)
    if min(p.stat().st_mtime_ns for p in stage.outputs) < common_source.stat().st_mtime_ns:
        return False
    for path in stage.outputs:
        if path.suffix == ".json":
            try:
                with path.open(encoding="utf-8") as handle:
                    json.load(handle)
            except (ValueError, OSError, UnicodeError):
                return False
    return True


def run_stage(stage: Stage, log_handle, *, force: bool = False) -> None:
    from src.resource_monitor import has_stage_resource

    resource_missing = stage.resource_required and not has_stage_resource(stage.label)
    if not force and not stage.always_run and stage_done(stage) and not resource_missing:
        line = f"[SKIP] {stage.label}"
        print(line)
        print(line, file=log_handle, flush=True)
        return
    if resource_missing and stage_done(stage) and not force:
        line = f"[RESOURCE REFRESH] {stage.label}"
        print(line)
        print(line, file=log_handle, flush=True)

    command = [sys.executable, "-m", stage.module, *stage.args]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TQDM_DISABLE", "1")
    # Off-the-shelf NER inference runs on CPU. Fine-tuning runs separately
    # in the parent process and may use CUDA; Ollama selects its own device.
    if stage.module in {"src.biored_models", "src.scispacy_bc5cdr", "src.biobert_bc5cdr", "src.pubmed_bc5cdr", "src.clinicalbert_bc5cdr", "src.d4data_bc5cdr"}:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["CAWV_NER_CPU_ONLY"] = "1"
    if stage.env:
        env.update(stage.env)

    header = f"{'=' * 78}\nRUNNING: {stage.label}\nCMD: {' '.join(command)}\n{'=' * 78}"
    print("\n" + header)
    print("\n" + header, file=log_handle, flush=True)

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    from src.resource_monitor import ResourceMonitor, append_stage_resource, folder_size_mb

    monitor = ResourceMonitor(
        stage.label,
        root_pid=process.pid,
        include_ollama=("medgemma" in stage.label.lower()),
        metadata={
            "module": stage.module,
            "command": " ".join(command),
        },
    ).start()

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        print(line, end="", file=log_handle, flush=True)
    return_code = process.wait()
    record = monitor.stop()
    record["return_code"] = int(return_code)
    record["expected_output_size_mb"] = float(
        sum(folder_size_mb(path) for path in stage.outputs if path.exists())
    )
    append_stage_resource(record)

    if return_code != 0:
        raise RuntimeError(f"Stage failed ({return_code}): {stage.label}\nModule: {stage.module}\nSee the current run log for the exception.")

    missing = [str(path) for path in stage.outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Stage completed but expected output(s) are missing: {stage.label}\n  "
            + "\n  ".join(missing)
        )


def check_raw_inputs() -> None:
    required = (
        ROOT / "data" / "raw" / "BIORED.zip",
        ROOT / "data" / "raw" / "bc5cdr" / "CDR_TrainingSet.PubTator.txt",
        ROOT / "data" / "raw" / "bc5cdr" / "CDR_DevelopmentSet.PubTator.txt",
        ROOT / "data" / "raw" / "bc5cdr" / "CDR_TestSet.PubTator.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raw input(s):\n  " + "\n  ".join(missing))


def check_ollama() -> None:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. Start Ollama and rerun."
        ) from exc

    installed = {
        str(item.get("name") or item.get("model") or "")
        for item in payload.get("models", [])
    }
    if MEDGEMMA_MODEL not in installed:
        raise RuntimeError(
            f"Required model is not installed: {MEDGEMMA_MODEL}\n"
            f"Run once: ollama pull {MEDGEMMA_MODEL}"
        )


def ensure_paths(paths: list[Path], message: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(message + "\n  " + "\n  ".join(missing))
