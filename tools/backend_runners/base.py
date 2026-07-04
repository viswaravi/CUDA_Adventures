#!/usr/bin/env python3
"""Shared backend-runner primitives.

Backend runners convert an ExperimentCase into an argv list. They do not run
profilers, parse YAML, or know kernel implementation details beyond the CLI
contract exposed by each backend executable.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from experiment_schema import ExperimentCase


class BackendRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class BackendCommand:
    executable: str
    args: list[str]
    launcher: list[str]
    display_backend: str

    def argv(self) -> list[str]:
        return [*self.launcher, self.executable, *self.args]


class BackendRunner:
    backend: str

    def __init__(self, executable: str | None):
        self.executable = executable

    def _require_executable(self) -> str:
        return self._check_executable(self.executable)

    def _resolve_executable(self, case: ExperimentCase) -> str:
        return self._check_executable(self.executable or case.executable)

    def _check_executable(self, executable: str | None) -> str:
        if not executable:
            raise BackendRunnerError(
                f'Backend runner executable not found: backend={self.backend}')
        path = pathlib.Path(executable)
        if not path.exists() and '/' in executable:
            raise BackendRunnerError(
                f'Backend runner executable not found: {executable}')
        return executable

    def build_command(self, case: ExperimentCase) -> BackendCommand:
        raise NotImplementedError


def bool_arg(value: bool) -> str:
    return 'true' if value else 'false'


def extend_scalar_args(command: list[str], values: dict[str, Any]) -> None:
    for key in sorted(values.keys()):
        value = values[key]
        if isinstance(value, (dict, list)) or value is None:
            continue
        command.extend([f'--{str(key).replace("_", "-")}', str(value)])
