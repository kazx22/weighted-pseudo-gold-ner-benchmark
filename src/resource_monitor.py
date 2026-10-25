from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

from src.pipeline_common import ROOT

RESOURCE_DIR = ROOT / "results" / "resource_usage"
JSONL_PATH = RESOURCE_DIR / "stage_resource_usage.jsonl"
CSV_PATH = RESOURCE_DIR / "stage_resource_usage.csv"


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _run_nvidia_smi(query: str):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            return []
        rows = []
        for line in result.stdout.splitlines():
            values = [item.strip() for item in line.split(",")]
            rows.append(values)
        return rows
    except Exception:
        return []


def _gpu_process_memory_mb(pids: set[int]) -> float | None:
    if not pids:
        return 0.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            return None
        total = 0.0
        found = False
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                mem = float(parts[1])
            except Exception:
                continue
            if pid in pids:
                total += mem
                found = True
        return total if found else 0.0
    except Exception:
        return None


def _gpu_system_sample():
    rows = _run_nvidia_smi("memory.used,utilization.gpu,power.draw")
    if not rows:
        return None
    memory = 0.0
    utils = []
    power = 0.0
    power_found = False
    for row in rows:
        if len(row) >= 1:
            value = _safe_float(row[0])
            if value is not None:
                memory += value
        if len(row) >= 2:
            value = _safe_float(row[1])
            if value is not None:
                utils.append(value)
        if len(row) >= 3:
            value = _safe_float(row[2])
            if value is not None:
                power += value
                power_found = True
    return {
        "memory_used_mb": memory,
        "utilization_percent": sum(utils) / len(utils) if utils else None,
        "power_watts": power if power_found else None,
    }


def _ollama_pids() -> set[int]:
    if psutil is None:
        return set()
    found = set()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (process.info.get("name") or "").lower()
            cmdline = " ".join(process.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmdline:
                found.add(int(process.info["pid"]))
        except Exception:
            continue
    return found


def _process_tree_pids(root_pid: int | None) -> set[int]:
    if root_pid is None:
        return set()
    pids = {int(root_pid)}
    if psutil is None:
        return pids
    try:
        root = psutil.Process(int(root_pid))
        for child in root.children(recursive=True):
            pids.add(int(child.pid))
    except Exception:
        pass
    return pids


def _rss_mb(pids: set[int]) -> float | None:
    if psutil is None:
        return None
    total = 0
    any_found = False
    for pid in pids:
        try:
            total += int(psutil.Process(pid).memory_info().rss)
            any_found = True
        except Exception:
            continue
    if not any_found:
        return 0.0
    return total / (1024 ** 2)


def folder_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    if path.is_file():
        try:
            total = path.stat().st_size
        except OSError:
            total = 0
    else:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    return total / (1024 ** 2)


class ResourceMonitor:
    """Best-effort stage telemetry.

    RAM is process-tree RSS when psutil is available.
    Process GPU memory comes from nvidia-smi compute-process accounting.
    Total GPU memory/utilisation/power are system-level GPU telemetry and may include
    unrelated GPU work. Energy is therefore an approximate whole-GPU reading, not a
    process-isolated energy measurement.
    """

    def __init__(
        self,
        label: str,
        *,
        root_pid: int | None = None,
        include_ollama: bool = False,
        poll_seconds: float = 1.0,
        metadata: dict | None = None,
    ):
        self.label = label
        self.root_pid = root_pid
        self.include_ollama = include_ollama
        self.poll_seconds = max(0.25, float(poll_seconds))
        self.metadata = metadata or {}
        self._stop = threading.Event()
        self._thread = None
        self._started = None
        self._last_sample_time = None
        self._energy_wh = 0.0
        self._gpu_util_samples = []
        self._peak_ram_mb = None
        self._peak_process_gpu_mb = None
        self._peak_total_gpu_mb = None

    def start(self):
        self._started = time.perf_counter()
        self._last_sample_time = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            pids = _process_tree_pids(self.root_pid)
            if self.include_ollama:
                pids.update(_ollama_pids())

            ram = _rss_mb(pids)
            if ram is not None:
                self._peak_ram_mb = ram if self._peak_ram_mb is None else max(self._peak_ram_mb, ram)

            process_gpu = _gpu_process_memory_mb(pids)
            if process_gpu is not None:
                self._peak_process_gpu_mb = (
                    process_gpu
                    if self._peak_process_gpu_mb is None
                    else max(self._peak_process_gpu_mb, process_gpu)
                )

            system_gpu = _gpu_system_sample()
            if system_gpu:
                total_gpu = system_gpu.get("memory_used_mb")
                if total_gpu is not None:
                    self._peak_total_gpu_mb = (
                        total_gpu
                        if self._peak_total_gpu_mb is None
                        else max(self._peak_total_gpu_mb, total_gpu)
                    )
                util = system_gpu.get("utilization_percent")
                if util is not None:
                    self._gpu_util_samples.append(float(util))
                power = system_gpu.get("power_watts")
                if power is not None and self._last_sample_time is not None:
                    elapsed_hours = max(0.0, now - self._last_sample_time) / 3600.0
                    self._energy_wh += float(power) * elapsed_hours
            self._last_sample_time = now
            self._stop.wait(self.poll_seconds)

    def stop(self) -> dict:
        if self._started is None:
            self.start()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_seconds * 3))
        wall_seconds = time.perf_counter() - float(self._started)
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": self.label,
            "wall_seconds": float(wall_seconds),
            "gpu_hours": float(wall_seconds / 3600.0),
            "peak_process_tree_ram_mb": self._peak_ram_mb,
            "peak_process_gpu_vram_mb": self._peak_process_gpu_mb,
            "peak_total_gpu_memory_used_mb": self._peak_total_gpu_mb,
            "mean_system_gpu_utilization_percent": (
                sum(self._gpu_util_samples) / len(self._gpu_util_samples)
                if self._gpu_util_samples
                else None
            ),
            "approx_system_gpu_energy_wh": (
                float(self._energy_wh) if self._energy_wh > 0 else None
            ),
            "ram_measurement": (
                "process-tree RSS" if psutil is not None else "not available; psutil missing"
            ),
            "gpu_process_measurement": "nvidia-smi compute-process memory when available",
            "gpu_system_measurement": "system-level nvidia-smi telemetry; not process-isolated",
            **self.metadata,
        }


def latest_stage_records() -> dict[str, dict]:
    if not JSONL_PATH.exists():
        return {}
    latest = {}
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        label = str(row.get("label", ""))
        if label:
            latest[label] = row
    return latest


def has_stage_resource(label: str) -> bool:
    row = latest_stage_records().get(label)
    if not row:
        return False
    try:
        return int(row.get("return_code", 0)) == 0
    except Exception:
        return False


def append_stage_resource(record: dict) -> None:
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    export_stage_csv()


def export_stage_csv() -> None:
    rows = list(latest_stage_records().values())
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
