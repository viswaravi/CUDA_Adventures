#!/usr/bin/env python3
"""cuDNN backend command builder placeholder for future non-vector-add ops."""

from __future__ import annotations

from experiment_schema import ExperimentCase

from .base import BackendCommand, BackendRunner, BackendRunnerError


class CudnnRunner(BackendRunner):
    backend = 'cudnn'

    def build_command(self, case: ExperimentCase) -> BackendCommand:
        if case.op == 'vector_add':
            raise BackendRunnerError("cuDNN is not a supported backend for vector_add")
        raise BackendRunnerError(
            f"cuDNN runner has no command mapping for op '{case.op}' yet")
