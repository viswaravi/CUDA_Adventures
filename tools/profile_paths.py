#!/usr/bin/env python3
"""Shared profiling output path helpers."""

from __future__ import annotations

import pathlib

from experiment_schema import sanitize_id


def profile_case_dir(
    output_root: pathlib.Path,
    experiment: str,
    tool: str,
    case_id: str,
) -> pathlib.Path:
    """Return the canonical output directory for one profiled experiment case."""
    parts = [part for part in sanitize_id(case_id).split('_') if part]
    return output_root / sanitize_id(experiment) / sanitize_id(tool) / pathlib.Path(*parts)
