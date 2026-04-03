"""Local hardware feasibility chain for experiment planning."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field


@dataclass(slots=True)
class HardwareProfile:
    os_name: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str | None
    vram_gb: float | None


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    objective: str
    min_cpu_cores: int = 2
    min_ram_gb: float = 4.0
    min_vram_gb: float = 0.0
    estimated_runtime_minutes: int = 10
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeasibilityReport:
    feasible: bool
    score: float
    reasons: list[str]
    mitigations: list[str]


def detect_local_hardware() -> HardwareProfile:
    cpu_cores = os.cpu_count() or 1
    ram_gb = _detect_ram_gb()
    gpu_name, vram_gb = _detect_gpu()
    return HardwareProfile(
        os_name=platform.system(),
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
    )


def is_hardware_feasible(experiment: ExperimentSpec, hardware: HardwareProfile | None = None) -> FeasibilityReport:
    """Score whether local hardware can execute an experiment."""
    hw = hardware or detect_local_hardware()
    reasons: list[str] = []
    mitigations: list[str] = []

    cpu_ratio = hw.cpu_cores / max(experiment.min_cpu_cores, 1)
    ram_ratio = hw.ram_gb / max(experiment.min_ram_gb, 0.1)

    if experiment.min_vram_gb > 0:
        vram_available = hw.vram_gb or 0.0
        vram_ratio = vram_available / experiment.min_vram_gb
    else:
        vram_ratio = 1.0

    score = round(min(cpu_ratio, ram_ratio, vram_ratio), 3)
    feasible = score >= 1.0

    if cpu_ratio < 1.0:
        reasons.append(f"CPU cores insufficient ({hw.cpu_cores} < {experiment.min_cpu_cores}).")
        mitigations.append("Reduce parallelism or use smaller batch sizes.")

    if ram_ratio < 1.0:
        reasons.append(f"RAM insufficient ({hw.ram_gb:.1f}GB < {experiment.min_ram_gb:.1f}GB).")
        mitigations.append("Use data sampling, chunked loading, or simpler models.")

    if vram_ratio < 1.0:
        reasons.append(
            f"GPU VRAM insufficient ({(hw.vram_gb or 0.0):.1f}GB < {experiment.min_vram_gb:.1f}GB)."
        )
        mitigations.append("Switch to CPU/quantized model or reduce sequence/batch lengths.")

    if feasible:
        reasons.append("Local hardware meets minimum experiment requirements.")
        mitigations.append("Optionally run pilot with 10-20% data first for validation.")

    return FeasibilityReport(feasible=feasible, score=score, reasons=reasons, mitigations=mitigations)


def autoimpute_experiment_specs(hypothesis: str) -> list[ExperimentSpec]:
    """Infer a practical local experiment set from a hypothesis."""
    lower = hypothesis.lower()
    experiments: list[ExperimentSpec] = []

    experiments.append(
        ExperimentSpec(
            name="eda_baseline",
            objective="Validate signal quality and feature coverage.",
            min_cpu_cores=2,
            min_ram_gb=4,
            estimated_runtime_minutes=5,
            tags=["eda", "baseline"],
        )
    )

    if any(k in lower for k in ["classif", "churn", "fraud", "segment"]):
        experiments.extend(
            [
                ExperimentSpec(
                    name="logistic_regression_baseline",
                    objective="Test linear separability baseline.",
                    min_cpu_cores=2,
                    min_ram_gb=8,
                    estimated_runtime_minutes=15,
                    tags=["classification", "baseline"],
                ),
                ExperimentSpec(
                    name="gradient_boosted_trees",
                    objective="Test non-linear interactions with tabular data.",
                    min_cpu_cores=4,
                    min_ram_gb=12,
                    estimated_runtime_minutes=30,
                    tags=["classification", "automl_candidate"],
                ),
            ]
        )

    elif any(k in lower for k in ["forecast", "demand", "timeseries", "regress", "price"]):
        experiments.extend(
            [
                ExperimentSpec(
                    name="ridge_regression_baseline",
                    objective="Test linear trend with regularization.",
                    min_cpu_cores=2,
                    min_ram_gb=8,
                    estimated_runtime_minutes=15,
                    tags=["regression", "baseline"],
                ),
                ExperimentSpec(
                    name="xgboost_regressor",
                    objective="Model non-linear effects for stronger performance.",
                    min_cpu_cores=4,
                    min_ram_gb=12,
                    estimated_runtime_minutes=40,
                    tags=["regression", "automl_candidate"],
                ),
            ]
        )

    if any(k in lower for k in ["llm", "finetune", "transformer"]):
        experiments.append(
            ExperimentSpec(
                name="qlora_finetune_trial",
                objective="Parameter-efficient fine-tuning pilot.",
                min_cpu_cores=8,
                min_ram_gb=24,
                min_vram_gb=16,
                estimated_runtime_minutes=120,
                tags=["llm", "gpu"],
            )
        )

    return experiments


def generate_feasibility_chain(hypothesis: str, hardware: HardwareProfile | None = None) -> list[tuple[ExperimentSpec, FeasibilityReport]]:
    specs = autoimpute_experiment_specs(hypothesis)
    return [(spec, is_hardware_feasible(spec, hardware=hardware)) for spec in specs]


def _detect_ram_gb() -> float:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return round((page_size * pages) / (1024**3), 2)
    except Exception:
        pass
    return 8.0


def _detect_gpu() -> tuple[str | None, float | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None, None
        first_line = proc.stdout.strip().splitlines()[0]
        name, mem = [p.strip() for p in first_line.split(",", maxsplit=1)]
        return name, round(float(mem) / 1024.0, 2)
    except Exception:
        return None, None
