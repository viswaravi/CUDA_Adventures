#!/usr/bin/env python3
"""CUDA backend command builder for normalized experiment cases."""

from __future__ import annotations

from experiment_schema import ExperimentCase

from .base import BackendCommand, BackendRunner, BackendRunnerError, bool_arg


class CudaRunner(BackendRunner):
    backend = 'cuda'

    def build_command(self, case: ExperimentCase) -> BackendCommand:
        executable = self._resolve_executable(case)
        if case.op == 'vector_add':
            args = self._vector_add_args(case)
        elif case.op in ('reduce', 'prefix_sum'):
            args = self._one_dim_args(case)
        elif case.op == 'matmul':
            args = self._matmul_args(case)
        elif case.op in ('rotation', 'conv2d'):
            args = self._image_op_args(case)
        else:
            raise BackendRunnerError(
                f"CUDA runner has no command mapping for op '{case.op}'")

        return BackendCommand(
            executable=executable,
            args=args,
            launcher=[],
            display_backend=self.backend,
        )

    @staticmethod
    def _vector_add_args(case: ExperimentCase) -> list[str]:
        if case.n is None:
            raise BackendRunnerError('Missing required shape.n for vector_add')
        if case.memory_host is None:
            raise BackendRunnerError('Missing required memory.host for vector_add cuda')
        if case.execution_mode is None:
            raise BackendRunnerError('Missing required execution.mode for vector_add cuda')

        args = [
            '--op', case.op,
            '--kernel', case.kernel,
            '--dtype', case.dtype,
            '--n', str(case.n),
            '--memory-host', case.memory_host,
            '--execution-mode', case.execution_mode,
            '--validate', bool_arg(case.validate),
            '--warmup', str(case.warmup),
            '--repeats', str(case.repeats),
        ]

        if case.execution_mode in ('chunked', 'chunked_multistream'):
            threshold = case.free_mem_threshold if case.free_mem_threshold is not None else 0.2
            args.extend(['--free-mem-threshold', str(threshold)])
        if case.execution_mode == 'chunked_multistream':
            args.extend(['--streams', str(case.streams or 1)])
        if case.tuning.get('vector_width') is not None:
            args.extend(['--vector-width', str(case.tuning['vector_width'])])

        return args

    @staticmethod
    def _one_dim_args(case: ExperimentCase) -> list[str]:
        if case.n is None:
            raise BackendRunnerError(f"Missing required shape.n for {case.op}")
        return [
            '--op', case.op,
            '--kernel', case.kernel,
            '--n', str(case.n),
        ]

    @staticmethod
    def _matmul_args(case: ExperimentCase) -> list[str]:
        args = [
            '--op', case.op,
            '--kernel', case.kernel,
        ]
        for field in ('m', 'n', 'k'):
            value = case.shape.get(field, case.args.get(field))
            if value is None:
                raise BackendRunnerError(f"Missing required shape.{field} for matmul")
            args.extend([f'--{field}', str(value)])
        return args

    @staticmethod
    def _image_op_args(case: ExperimentCase) -> list[str]:
        args = [
            '--op', case.op,
            '--kernel', case.kernel,
        ]
        for key, value in case.params.items():
            args.extend([f"--{str(key).replace('_', '-')}", str(value)])
        for key, value in case.args.items():
            args.extend([f"--{str(key).replace('_', '-')}", str(value)])
        return args
