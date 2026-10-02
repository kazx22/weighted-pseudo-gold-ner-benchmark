
from __future__ import annotations

import json
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
    return bool(stage.outputs) and all(path.exists() for path in stage.outputs)


def run_stage(stage: Stage, log_handle, *, force: bool = False) -> None:
    if not force and stage_done(stage):
        line = f"[SKIP] {stage.label}"
        print(line)
        print(line, file=log_handle, flush=True)
        return

    command = [sys.executable, "-m", stage.module, *stage.args]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
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
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        print(line, end="", file=log_handle, flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Stage failed ({return_code}): {stage.label}")

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
