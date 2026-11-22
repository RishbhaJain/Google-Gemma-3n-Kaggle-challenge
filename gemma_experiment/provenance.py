"""Content-free runtime provenance for training and evaluation artifacts."""

from __future__ import annotations

import platform
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

RUNTIME_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "peft",
    "transformers",
    "trl",
    "unsloth",
)


def _package_versions(
    package_names: tuple[str, ...],
    version_lookup: Callable[[str], str],
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = version_lookup(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def capture_runtime_environment(
    torch: Any,
    *,
    cuda_required: bool | None = None,
    package_names: tuple[str, ...] = RUNTIME_DISTRIBUTIONS,
    version_lookup: Callable[[str], str] = version,
) -> dict[str, object]:
    """Capture the software and accelerator context needed to audit a run."""

    cuda_available = bool(torch.cuda.is_available())
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": _package_versions(package_names, version_lookup),
        "gpu": None,
        "gpu_total_memory_gib": None,
        "gpu_compute_capability": None,
    }
    if cuda_required is not None:
        environment["cuda_required"] = cuda_required
    if not cuda_available:
        return environment

    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    environment.update(
        {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_gib": properties.total_memory / 1024**3,
            "gpu_compute_capability": f"{major}.{minor}",
        }
    )
    return environment
