#!/usr/bin/env python3
"""cuBLAS backend command builder for normalized experiment cases."""

from __future__ import annotations

from experiment_schema import ExperimentCase

from .base import BackendCommand, BackendRunner, BackendRunnerError, bool_arg


class CublasRunner(BackendRunner):
    backend = 'cublas'

    def build_command(self, case: ExperimentCase) -> BackendCommand:
        executable = self._resolve_executable(case)
        if case.op != 'vector_add':
            raise BackendRunnerError(
                f"cuBLAS runner has no command mapping for op '{case.op}'")
        if case.n is None:
            raise BackendRunnerError('Missing required shape.n for cublas vector_add')

        args = [
            '--op', case.op,
            '--kernel', case.kernel,
            '--dtype', case.dtype,
            '--n', str(case.n),
            '--validate', bool_arg(case.validate),
            '--warmup', str(case.warmup),
            '--repeats', str(case.repeats),
        ]
        if case.params.get('alpha') is not None:
            args.extend(['--alpha', str(case.params['alpha'])])

        return BackendCommand(
            executable=executable,
            args=args,
            launcher=[],
            display_backend=self.backend,
        )
