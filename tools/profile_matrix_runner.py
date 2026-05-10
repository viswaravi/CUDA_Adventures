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


def load_matrix(matrix_path: pathlib.Path) -> Tuple[str, List[Dict[str, Any]]]:
    with matrix_path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh)

    if isinstance(data, list):
        return matrix_path.stem, data
    if isinstance(data, dict):
        experiment_name = str(data.get('experiment', matrix_path.stem))
        for key in ('sweeps', 'experiments', 'cases', 'variants'):
            value = data.get(key)
            if isinstance(value, list):
                return experiment_name, value
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

    profile_runner = pathlib.Path(args.profile_runner) if args.profile_runner else pathlib.Path(
        __file__).with_name('profile_runner.py')
    experiment_name, entries = load_matrix(matrix_path)

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
