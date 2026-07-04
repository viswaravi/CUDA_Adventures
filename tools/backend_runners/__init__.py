#!/usr/bin/env python3
"""Backend runner registry used by matrix_runner.py and profile_runner.py."""

from __future__ import annotations

from experiment_schema import ExperimentCase

from .base import BackendCommand, BackendRunner, BackendRunnerError
from .cublas import CublasRunner
from .cuda import CudaRunner
from .cudnn import CudnnRunner
from .triton import TritonRunner


def make_backend_runners(
    *,
    binary: str | None = None,
    cuda_binary: str | None = None,
    triton_binary: str | None = None,
    cublas_binary: str | None = None,
    cudnn_binary: str | None = None,
) -> dict[str, BackendRunner]:
    return {
        'cuda': CudaRunner(cuda_binary or binary),
        'triton': TritonRunner(triton_binary or binary),
        'cublas': CublasRunner(cublas_binary or binary),
        'cudnn': CudnnRunner(cudnn_binary or binary),
    }


def build_backend_command(
    case: ExperimentCase,
    runners: dict[str, BackendRunner],
) -> BackendCommand:
    runner = runners.get(case.backend)
    if runner is None:
        raise BackendRunnerError(
            f"Unsupported backend '{case.backend}' for op '{case.op}'")
    return runner.build_command(case)
