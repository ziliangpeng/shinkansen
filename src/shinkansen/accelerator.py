"""Detect accelerator backend (CUDA vs ROCm) at runtime.

PyTorch ROCm builds expose `torch.cuda.*` for AMD too, with `torch.version.hip`
set to the HIP version string. NVIDIA builds have `torch.version.cuda` set and
`torch.version.hip` is None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Backend = Literal["cuda", "rocm", "cpu"]


@dataclass(frozen=True)
class AcceleratorInfo:
    backend: Backend
    device_count: int
    device_name: str | None
    arch: str | None  # e.g. "sm_90" (H100), "gfx942" (MI300X / MI325X)
    torch_version: str
    runtime_version: str | None  # CUDA or HIP version

    @property
    def is_rocm(self) -> bool:
        return self.backend == "rocm"

    @property
    def is_cuda(self) -> bool:
        return self.backend == "cuda"


def detect() -> AcceleratorInfo:
    if not torch.cuda.is_available():
        return AcceleratorInfo(
            backend="cpu",
            device_count=0,
            device_name=None,
            arch=None,
            torch_version=torch.__version__,
            runtime_version=None,
        )

    is_rocm = torch.version.hip is not None
    backend: Backend = "rocm" if is_rocm else "cuda"
    runtime = torch.version.hip if is_rocm else torch.version.cuda

    idx = 0
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    if is_rocm:
        arch = getattr(props, "gcnArchName", None)
    else:
        arch = f"sm_{props.major}{props.minor}"

    return AcceleratorInfo(
        backend=backend,
        device_count=torch.cuda.device_count(),
        device_name=name,
        arch=arch,
        torch_version=torch.__version__,
        runtime_version=runtime,
    )


def summary() -> str:
    info = detect()
    if info.backend == "cpu":
        return f"shinkansen: CPU only (torch {info.torch_version})"
    return (
        f"shinkansen: {info.backend.upper()} | "
        f"{info.device_count}x {info.device_name} ({info.arch}) | "
        f"torch {info.torch_version} / {info.backend} {info.runtime_version}"
    )
