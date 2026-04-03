from stella_ml.feasibility import (
    ExperimentSpec,
    HardwareProfile,
    autoimpute_experiment_specs,
    is_hardware_feasible,
)
from stella_ml.harness import OpenClawStyleHarness


def test_is_hardware_feasible_true_case() -> None:
    hardware = HardwareProfile(os_name="Linux", cpu_cores=16, ram_gb=64, gpu_name="RTX", vram_gb=24)
    exp = ExperimentSpec(name="x", objective="y", min_cpu_cores=4, min_ram_gb=8, min_vram_gb=0)
    report = is_hardware_feasible(exp, hardware)
    assert report.feasible is True
    assert report.score >= 1.0


def test_autoimpute_experiment_specs_for_regression() -> None:
    specs = autoimpute_experiment_specs("Forecast demand with regression")
    names = {s.name for s in specs}
    assert "eda_baseline" in names
    assert "ridge_regression_baseline" in names


def test_harness_isHardwareFeasible_chain() -> None:
    harness = OpenClawStyleHarness()
    hardware = HardwareProfile(os_name="Linux", cpu_cores=2, ram_gb=4, gpu_name=None, vram_gb=None)
    chain = harness.isHardwareFeasible("LLM finetune experiment", hardware)
    assert chain
    assert any(name == "qlora_finetune_trial" for name, _ in chain)
