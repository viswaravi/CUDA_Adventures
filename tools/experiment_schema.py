#!/usr/bin/env python3
"""Schema loader for normalized CUDA kernel lab experiments.

Experiment YAML describes op/backend/kernel/dtype plus structured shape,
memory, execution, tuning, and params blocks. This module validates that data
and returns immutable ExperimentCase objects for the runners.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentCase:
    id: str
    group: str | None
    op: str
    backend: str
    kernel: str
    dtype: str
    validate: bool
    warmup: int
    repeats: int
    args: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    shape: dict[str, Any] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    profilers: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    executable: str | None = None

    @property
    def memory_host(self) -> str | None:
        value = self.memory.get('host')
        return str(value) if value is not None else None

    @property
    def execution_mode(self) -> str | None:
        value = self.execution.get('mode')
        return str(value) if value is not None else None

    @property
    def streams(self) -> int | None:
        value = self.execution.get('streams')
        return _as_int(value, 'execution.streams') if value is not None else None

    @property
    def chunked(self) -> bool | None:
        value = self.execution.get('chunked')
        return _as_bool(value) if value is not None else None

    @property
    def n(self) -> int | None:
        value = self.shape.get('n', self.args.get('n'))
        return _as_int(value, 'shape.n') if value is not None else None

    @property
    def free_mem_threshold(self) -> float | None:
        value = self.execution.get(
            'free_mem_threshold',
            self.execution.get('free-mem-threshold', self.args.get('free-mem-threshold')),
        )
        return _float_or_none(value, 'execution.free_mem_threshold')


@dataclass(frozen=True)
class ExperimentMatrix:
    experiment: str
    profilers: dict[str, dict[str, Any]]
    backend_capabilities: dict[str, dict[str, Any]]
    cases: list[ExperimentCase]


def sanitize_id(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace('.', 'p')
    text = text.replace('/', '_')
    text = re.sub(r'[^a-z0-9_+-]+', '_', text)
    text = text.replace('-', '_')
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or 'case'


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f'{name} must be a YAML object')


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {name}: must be an integer') from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ('1', 'true', 'yes', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'off'):
            return False
    return bool(value)


def _float_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {name}: must be a number') from exc


def _normalize_keyed_args(raw_args: Any) -> dict[str, Any]:
    args = _as_dict(raw_args, 'args')
    return {str(key).replace('_', '-'): value for key, value in args.items()}


def _normalize_capabilities(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('backend_capabilities must be a YAML object')

    capabilities: dict[str, dict[str, Any]] = {}
    for backend, config in raw.items():
        backend_name = str(backend)
        config_dict = _as_dict(config, f'backend_capabilities.{backend_name}')
        capabilities[backend_name] = {}
        for field_name in ('kernels', 'memory', 'execution', 'dtypes'):
            values = config_dict.get(field_name, [])
            if not isinstance(values, list):
                raise ValueError(
                    f'backend_capabilities.{backend_name}.{field_name} must be a list')
            capabilities[backend_name][field_name] = [str(item) for item in values]
        if config_dict.get('executable') is not None:
            capabilities[backend_name]['executable'] = str(config_dict['executable'])
    return capabilities


def normalize_sweep(
    raw: dict[str, Any],
    defaults: dict[str, Any],
    capabilities: dict[str, dict[str, Any]] | None = None,
) -> ExperimentCase:
    if not isinstance(raw, dict):
        raise ValueError(f'Sweep entry must be a YAML object: {raw!r}')
    if 'variant' in raw:
        raise ValueError(
            f"Legacy 'variant' field is not supported in normalized experiments: {raw!r}")

    args = _normalize_keyed_args(raw.get('args'))

    op = str(raw.get('op', defaults.get('op', 'unknown')))
    backend = str(raw.get('backend', defaults.get('backend', 'cuda')))
    backend_caps = (capabilities or {}).get(backend, {})
    kernel = str(raw.get('kernel', defaults.get('kernel', 'scalar')))
    dtype = str(raw.get('dtype', defaults.get('dtype', 'float32')))
    validate = _as_bool(raw.get('validate', defaults.get('validate', True)))
    warmup = _as_int(raw.get('warmup', defaults.get('warmup', 0)), 'warmup')
    repeats = _as_int(raw.get('repeats', defaults.get('repeats', 1)), 'repeats')
    group = str(raw['group']) if raw.get('group') is not None else None
    memory = _as_dict(raw.get('memory'), 'memory')
    execution = _as_dict(raw.get('execution'), 'execution')
    shape = _as_dict(raw.get('shape'), 'shape')

    if 'host' not in memory and defaults.get('memory_host') is not None:
        memory['host'] = defaults['memory_host']
    if 'mode' not in execution and defaults.get('execution_mode') is not None:
        execution['mode'] = defaults['execution_mode']
    if raw.get('id') is None:
        raise ValueError(f"Normalized sweep is missing required 'id': {raw!r}")
    case_id = sanitize_id(str(raw['id']))

    case = ExperimentCase(
        id=case_id,
        group=group,
        op=op,
        backend=backend,
        kernel=kernel,
        dtype=dtype,
        validate=validate,
        warmup=warmup,
        repeats=repeats,
        args=args,
        memory=memory,
        execution=execution,
        shape=shape,
        tuning=_as_dict(raw.get('tuning'), 'tuning'),
        params=_as_dict(raw.get('params'), 'params'),
        profilers=_as_dict(raw.get('profilers'), 'profilers'),
        source=dict(raw),
        executable=(
            str(raw['executable'])
            if raw.get('executable') is not None
            else backend_caps.get('executable')
        ),
    )
    return case


def validate_case(
    case: ExperimentCase,
    capabilities: dict[str, dict[str, Any]],
) -> None:
    if case.backend not in capabilities:
        raise ValueError(
            f"Unsupported backend '{case.backend}' for op '{case.op}'")

    backend_caps = capabilities[case.backend]
    if case.kernel not in backend_caps.get('kernels', []):
        raise ValueError(
            f"Unsupported kernel '{case.kernel}' for backend '{case.backend}'")
    if backend_caps.get('dtypes') and case.dtype not in backend_caps.get('dtypes', []):
        raise ValueError(
            f"Unsupported dtype '{case.dtype}' for backend '{case.backend}'")
    if case.memory_host is not None and case.memory_host not in backend_caps.get('memory', []):
        raise ValueError(
            f"Unsupported memory host '{case.memory_host}' for backend '{case.backend}'")
    if case.execution_mode is not None and case.execution_mode not in backend_caps.get('execution', []):
        raise ValueError(
            f"Unsupported execution mode '{case.execution_mode}' for backend '{case.backend}'")
    if case.op == 'vector_add' and case.n is None:
        raise ValueError('Missing required shape.n for vector_add')
    if case.n is not None and case.n <= 0:
        raise ValueError('Invalid shape.n: must be > 0')
    if case.streams is not None and case.streams < 1:
        raise ValueError('Invalid execution.streams: must be >= 1')
    if case.free_mem_threshold is not None and not (0.0 < case.free_mem_threshold <= 1.0):
        raise ValueError(
            'Invalid execution.free_mem_threshold: must be in (0, 1]')


def load_experiment_matrix(path: pathlib.Path) -> ExperimentMatrix:
    with path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh)

    if isinstance(data, list):
        data = {'experiment': path.stem, 'sweeps': data}
    if not isinstance(data, dict):
        raise ValueError(f'Unsupported matrix format in {path}')

    experiment_name = str(data.get('experiment', path.stem))
    defaults = _as_dict(data.get('defaults'), 'defaults')
    defaults.setdefault('op', data.get('op', experiment_name))
    profilers = _as_dict(data.get('profilers'), 'profilers')
    capabilities = _normalize_capabilities(data.get('backend_capabilities'))

    raw_sweeps = None
    for key in ('sweeps', 'experiments', 'cases'):
        if isinstance(data.get(key), list):
            raw_sweeps = data[key]
            break
    if raw_sweeps is None:
        raise ValueError(f'Unsupported matrix format in {path}: missing sweeps list')

    cases = [normalize_sweep(raw, defaults, capabilities) for raw in raw_sweeps]
    if capabilities:
        for case in cases:
            validate_case(case, capabilities)

    return ExperimentMatrix(
        experiment=experiment_name,
        profilers=profilers,
        backend_capabilities=capabilities,
        cases=cases,
    )
