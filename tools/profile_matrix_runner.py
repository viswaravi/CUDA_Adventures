#!/usr/bin/env python3

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_PROFILER_BIN = {
    'ncu': '/usr/bin/ncu',
    'nsys': '/usr/bin/nsys',
}

VALID_TOOL_CONFIG_KEYS = {
    'ncu': {'replay-mode', 'profiler-set', 'sections', 'args'},
    'nsys': {'trace', 'sample', 'capture-range', 'args'},
}


def append_passthrough_option(command: List[str], option: str, value: str | None) -> None:
    if value:
        command.extend([option, value])


def build_profiler_runner_args(args: argparse.Namespace) -> List[str]:
    profiler_runner_args: List[str] = []
    for profiler_arg in args.profiler_args:
        profiler_runner_args.extend(['--profiler-arg', profiler_arg])

    append_passthrough_option(
        profiler_runner_args, '--ncu-replay-mode', args.ncu_replay_mode)
    append_passthrough_option(
        profiler_runner_args, '--ncu-profiler-set', args.ncu_profiler_set)
    for section in args.ncu_sections:
        profiler_runner_args.extend(['--ncu-section', section])

    append_passthrough_option(
        profiler_runner_args, '--nsys-trace', args.nsys_trace)
    append_passthrough_option(
        profiler_runner_args, '--nsys-sample', args.nsys_sample)
    append_passthrough_option(
        profiler_runner_args, '--nsys-capture-range', args.nsys_capture_range)
    return profiler_runner_args


def normalize_string_list(value: Any, field_name: str, context: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"{context}: '{field_name}' must be a YAML list")


def normalize_tool_config(tool: str, config: Any, context: str) -> Dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(
            f'{context}: profiler config for {tool!r} must be an object')

    normalized: Dict[str, Any] = {}
    for raw_key, value in config.items():
        key = str(raw_key).replace('_', '-')
        if key not in VALID_TOOL_CONFIG_KEYS[tool]:
            valid_keys = ', '.join(sorted(VALID_TOOL_CONFIG_KEYS[tool]))
            raise ValueError(
                f"{context}: unsupported {tool!r} profiler key {raw_key!r}. "
                f'Valid keys: {valid_keys}'
            )

        if key in ('sections', 'args'):
            normalized[key] = normalize_string_list(value, key, context)
        elif value is not None:
            normalized[key] = str(value)

    return normalized


def normalize_profiler_config(config: Any, context: str) -> Dict[str, Dict[str, Any]]:
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f'{context}: profilers must be a YAML object')

    normalized: Dict[str, Dict[str, Any]] = {}
    for tool in ('ncu', 'nsys'):
        tool_config = config.get(tool)
        if tool_config is not None:
            normalized[tool] = normalize_tool_config(
                tool, tool_config, context)
    return normalized


def merge_profiler_configs(
    base: Dict[str, Dict[str, Any]],
    override: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for tool in ('ncu', 'nsys'):
        tool_config: Dict[str, Any] = {}
        if tool in base:
            tool_config.update(base[tool])
        if tool in override:
            tool_config.update(override[tool])
        if tool_config:
            merged[tool] = tool_config
    return merged


def build_profiler_runner_args_from_config(
    tool: str,
    config: Dict[str, Dict[str, Any]],
) -> List[str]:
    tool_config = config.get(tool, {})
    profiler_runner_args: List[str] = []

    for profiler_arg in tool_config.get('args', []):
        profiler_runner_args.extend(['--profiler-arg', profiler_arg])

    if tool == 'ncu':
        append_passthrough_option(
            profiler_runner_args, '--ncu-replay-mode', tool_config.get('replay-mode'))
        append_passthrough_option(
            profiler_runner_args, '--ncu-profiler-set', tool_config.get('profiler-set'))
        for section in tool_config.get('sections', []):
            profiler_runner_args.extend(['--ncu-section', section])
    else:
        append_passthrough_option(
            profiler_runner_args, '--nsys-trace', tool_config.get('trace'))
        append_passthrough_option(
            profiler_runner_args, '--nsys-sample', tool_config.get('sample'))
        append_passthrough_option(
            profiler_runner_args, '--nsys-capture-range', tool_config.get('capture-range'))

    return profiler_runner_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Expand a YAML experiment matrix and run Nsight profiling for each entry.'
    )
    parser.add_argument('--matrix', required=True,
                        help='Path to module-local YAML sweep file')
    parser.add_argument('--tool', choices=['ncu', 'nsys'], required=True)
    parser.add_argument('--profiler-bin', default=None,
                        help='Optional path to profiler binary; defaults by --tool')
    parser.add_argument('--binary', required=True,
                        help='Path to the profiled binary')
    parser.add_argument('--output-dir', default='profiling',
                        help='Base output directory for reports (default: profiling)')
    parser.add_argument('--build-tag', default='dev',
                        help='Build tag used in report stems')
    parser.add_argument('--profile-runner', default=None,
                        help='Optional path to profile_runner.py')
    parser.add_argument(
        '--profiler-arg',
        dest='profiler_args',
        action='append',
        default=[],
        help='Repeatable raw profiler argument forwarded to profile_runner.py. Use = when the value starts with --.',
    )
    parser.add_argument('--ncu-replay-mode', default=None,
                        help='Optional NCU replay mode, forwarded as --replay-mode.')
    parser.add_argument('--ncu-profiler-set', default=None,
                        help='Optional NCU section set, forwarded as --set.')
    parser.add_argument('--ncu-section', dest='ncu_sections', action='append', default=[],
                        help='Repeatable NCU section, forwarded as --section.')
    parser.add_argument('--nsys-trace', default=None,
                        help='Optional NSYS trace domains, forwarded as --trace.')
    parser.add_argument('--nsys-sample', default=None,
                        help='Optional NSYS sampling mode, forwarded as --sample.')
    parser.add_argument('--nsys-capture-range', default=None,
                        help='Optional NSYS capture range, forwarded as --capture-range.')

    # Single-entry mode
    parser.add_argument('--label', default=None,
                        help='Run only the entry with this label (instead of the full sweep).')
    parser.add_argument('--variant', default=None,
                        help='Disambiguate when multiple entries share the same label.')
    return parser.parse_args()


def resolve_profiler_bin(tool: str, profiler_bin: str | None) -> str:
    if profiler_bin:
        return profiler_bin

    default_path = DEFAULT_PROFILER_BIN.get(tool)
    if default_path and pathlib.Path(default_path).exists():
        return default_path

    return tool


def load_matrix(
    matrix_path: pathlib.Path,
) -> Tuple[str, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    with matrix_path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh)

    if isinstance(data, list):
        return matrix_path.stem, {}, data
    if isinstance(data, dict):
        experiment_name = str(data.get('experiment', matrix_path.stem))
        profiler_config = normalize_profiler_config(
            data.get('profilers'), f'matrix {matrix_path}')
        for key in ('sweeps', 'experiments', 'cases', 'variants'):
            value = data.get(key)
            if isinstance(value, list):
                return experiment_name, profiler_config, value
    raise ValueError(f'Unsupported matrix format in {matrix_path}')


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace('.', 'p')
    text = text.replace('/', '-')
    text = re.sub(r'[^a-z0-9_-]+', '-', text)
    text = re.sub(r'-{2,}', '-', text)
    return text.strip('-_') or '0'


def build_report_stem(entry: Dict[str, Any], build_tag: str) -> str:
    parts: List[str] = [slugify(entry['variant'])]

    label = entry.get('label')
    if label:
        parts.append(slugify(label))

    args = entry.get('args', {})
    if not isinstance(args, dict):
        raise ValueError(
            f"Entry for variant '{entry.get('variant')}' must use an object for 'args'")

    for key in sorted(args.keys()):
        parts.append(f"{slugify(key)}{slugify(args[key])}")

    parts.append(f'build{slugify(build_tag)}')
    return '__'.join(parts)


def to_program_args(entry: Dict[str, Any]) -> List[str]:
    args = entry.get('args', {})
    if not isinstance(args, dict):
        raise ValueError(
            f"Entry for variant '{entry.get('variant')}' must use an object for 'args'")

    program_args: List[str] = ['--variant', str(entry['variant'])]
    for key in sorted(args.keys()):
        program_args.extend([f'--{key}', str(args[key])])
    return program_args


def main() -> int:
    args = parse_args()
    matrix_path = pathlib.Path(args.matrix)
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    profiler_bin = resolve_profiler_bin(args.tool, args.profiler_bin)
    profiler_runner_args = build_profiler_runner_args(args)

    profile_runner = pathlib.Path(args.profile_runner) if args.profile_runner else pathlib.Path(
        __file__).with_name('profile_runner.py')
    experiment_name, matrix_profiler_config, entries = load_matrix(matrix_path)

    if not entries:
        print(f'No experiment entries found in {matrix_path}')
        return 0

    # Filter to a single entry when --label is given
    if args.label:
        filtered = [
            e for e in entries
            if str(e.get('label', '')) == args.label
            and (args.variant is None or str(e.get('variant', '')) == args.variant)
        ]
        if not filtered:
            hint = f'label={args.label!r}' + \
                (f', variant={args.variant!r}' if args.variant else '')
            print(
                f'error: no entry found for {hint} in {matrix_path}', file=sys.stderr)
            return 2
        if len(filtered) > 1:
            names = [f"variant={e.get('variant')!r}" for e in filtered]
            print(
                f'error: multiple entries match label={args.label!r}: {", ".join(names)}. '
                'Use --variant to disambiguate.',
                file=sys.stderr,
            )
            return 2
        entries = filtered

    experiment_dir = output_root / slugify(experiment_name)
    tool_dir = experiment_dir / slugify(args.tool)

    for entry in entries:
        if 'variant' not in entry:
            raise ValueError(f"Matrix entry is missing 'variant': {entry}")
        entry_profiler_config = normalize_profiler_config(
            entry.get('profilers'),
            f"matrix entry variant={entry.get('variant')!r}, label={entry.get('label', 'default')!r}",
        )
        effective_profiler_config = merge_profiler_configs(
            matrix_profiler_config,
            entry_profiler_config,
        )
        effective_profiler_runner_args = [
            *build_profiler_runner_args_from_config(args.tool, effective_profiler_config),
            *profiler_runner_args,
        ]

        variant = slugify(entry['variant'])
        label = slugify(entry.get('label', 'default'))
        variant_dir = tool_dir / variant / label
        report_stem = build_report_stem(entry, args.build_tag)
        program_args = to_program_args(entry)

        command = [
            sys.executable,
            str(profile_runner),
            '--tool', args.tool,
            '--profiler-bin', profiler_bin,
            '--binary', args.binary,
            '--output-dir', str(variant_dir),
            '--report-stem', report_stem,
            *effective_profiler_runner_args,
            '--',
            *program_args,
        ]

        print('Running matrix entry:', ' '.join(program_args))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode

    return 0


if __name__ == '__main__':
    sys.exit(main())
