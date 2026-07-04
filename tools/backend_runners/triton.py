#!/usr/bin/env python3
"""Triton backend command builder for normalized experiment cases."""

from __future__ import annotations

import sys

from experiment_schema import ExperimentCase

from .base import BackendCommand, BackendRunner, BackendRunnerError, bool_arg


class TritonRunner(BackendRunner):
    backend = 'triton'

    def build_command(self, case: ExperimentCase) -> BackendCommand:
        executable = self._resolve_executable(case)
        if case.op != 'vector_add':
            raise BackendRunnerError(
                f"Triton runner has no command mapping for op '{case.op}'")
        if case.n is None:
            raise BackendRunnerError('Missing required shape.n for vector_add')

        args = [
            '--op', case.op,
            '--kernel', case.kernel,
            '--dtype', case.dtype,
            '--n', str(case.n),
            '--validate', bool_arg(case.validate),
            '--warmup', str(case.warmup),
            '--repeats', str(case.repeats),
        ]
        if case.tuning.get('block_size') is not None:
            args.extend(['--block-size', str(case.tuning['block_size'])])

        launcher = [sys.executable] if executable.endswith('.py') else []
        return BackendCommand(
            executable=executable,
            args=args,
            launcher=launcher,
            display_backend=self.backend,
        )
