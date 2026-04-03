"""Optional ML backend integrations for NLP, deep learning, probabilistic models, and AutoML ecosystems."""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BackendStatus:
    name: str
    available: bool
    version: str | None
    notes: str = ""


def detect_backend_availability() -> dict[str, BackendStatus]:
    """Check whether key libraries are installed and importable."""
    packages = {
        "nltk": "nltk",
        "spacy": "spacy",
        "tensorflow": "tensorflow",
        "keras": "keras",
        "torch": "torch",
        "sklearn": "sklearn",
        "pymc": "pymc",
    }

    status: dict[str, BackendStatus] = {}
    for alias, module_name in packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            status[alias] = BackendStatus(alias, True, str(version))
        except Exception as exc:
            status[alias] = BackendStatus(alias, False, None, notes=str(exc))
    return status


def install_packages(packages: list[str]) -> dict[str, bool]:
    """Install requested packages into current Python environment."""
    results: dict[str, bool] = {}
    for pkg in packages:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True,
            check=False,
        )
        results[pkg] = proc.returncode == 0
    return results


def demo_cuda() -> dict[str, Any]:
    """Return CUDA capability summary using PyTorch when available."""
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "message": "PyTorch not installed.",
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
        "message": "CUDA ready" if cuda_available else "CUDA not available; CPU mode active.",
    }


def recommend_cpu_sota_1bit_models() -> list[dict[str, str]]:
    """Recommend CPU-friendly low-bit model families when GPU is unavailable."""
    return [
        {
            "family": "BitNet b1.58-style",
            "use_case": "CPU-first LLM inference with extreme quantization",
            "runtime": "llama.cpp / custom kernels",
            "note": "Use quantized checkpoints and evaluate perplexity drift before production.",
        },
        {
            "family": "GGUF ultra-quantized models",
            "use_case": "Fast local inference on CPU-only machines",
            "runtime": "llama.cpp",
            "note": "Prefer task-tuned instruct variants for analysis workflows.",
        },
    ]


def list_sklearn_estimators() -> list[str]:
    """Expose scikit-learn estimator names when sklearn is installed."""
    try:
        from sklearn.utils import all_estimators
    except Exception:
        return []

    return sorted({name for name, _ in all_estimators()})


def create_tensorflow_mlp(input_dim: int, output_dim: int) -> Any:
    """Build a minimal Keras MLP architecture."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(output_dim),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_pytorch_mlp(input_dim: int, output_dim: int) -> Any:
    """Build a minimal PyTorch MLP architecture."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )


def run_pymc_linear_regression(x: list[float], y: list[float]) -> str:
    """Fit a tiny Bayesian linear regression with PyMC if available."""
    try:
        import pymc as pm
    except Exception as exc:
        return f"PyMC unavailable: {exc}"

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + beta * x
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        pm.sample(draws=200, tune=200, chains=1, progressbar=False)
    return "PyMC regression sampled successfully"
