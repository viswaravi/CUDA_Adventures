#!/usr/bin/env python3
"""Expand normalized experiment YAML into profiled backend runs.

This runner owns orchestration only: it loads cases, applies filters, computes
per-case output locations, and delegates single-case execution to
profile_runner.py.
"""

import argparse
import pathlib
import subprocess
import sys

from experiment_schema import ExperimentCase, load_experiment_matrix, sanitize_id
from profile_paths import profile_case_dir


def append_option(command: list[str], option: str, value: str | None) -> None:
    if value:
        command.extend([option, value])


def build_cli_override_args(args: argparse.Namespace) -> list[str]:
    """Return user-provided overrides forwarded to profile_runner.py."""
    overrides: list[str] = []

    append_option(overrides, '--profiler-bin', args.profiler_bin)
    append_option(overrides, '--binary', args.binary)
    append_option(overrides, '--cuda-binary', args.cuda_binary)
    append_option(overrides, '--triton-binary', args.triton_binary)
    append_option(overrides, '--cublas-binary', args.cublas_binary)
    append_option(overrides, '--cudnn-binary', args.cudnn_binary)

    for profiler_arg in args.profiler_args:
        overrides.extend(['--profiler-arg', profiler_arg])

    append_option(overrides, '--ncu-replay-mode', args.ncu_replay_mode)
    append_option(overrides, '--ncu-profiler-set', args.ncu_profiler_set)
    for section in args.ncu_sections:
        overrides.extend(['--ncu-section', section])

    append_option(overrides, '--nsys-trace', args.nsys_trace)
    append_option(overrides, '--nsys-sample', args.nsys_sample)
    append_option(overrides, '--nsys-capture-range', args.nsys_capture_range)
    return overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Expand a YAML experiment matrix and run Nsight profiling for each entry.'
    )
    parser.add_argument('--matrix', required=True,
                        help='Path to module-local YAML sweep file')
    parser.add_argument('--tool', choices=['ncu', 'nsys'], required=True)
    parser.add_argument('--profiler-bin', default=None,
                        help='Optional path to profiler binary; defaults by --tool')
    parser.add_argument('--binary', default=None,
                        help='Default backend executable for single-backend usage')
    parser.add_argument('--cuda-binary', default=None,
                        help='CUDA backend executable; defaults to --binary')
    parser.add_argument('--triton-binary', default=None,
                        help='Triton backend runner script; defaults to --binary')
    parser.add_argument('--cublas-binary', default=None,
                        help='cuBLAS backend executable; defaults to --binary')
    parser.add_argument('--cudnn-binary', default=None,
                        help='cuDNN backend executable; defaults to --binary')
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

    parser.add_argument('--id', dest='case_id', default=None,
                        help='Run only the normalized experiment case with this id.')
    parser.add_argument('--backend', default=None,
                        help='Run only cases for this backend.')
    parser.add_argument('--kernel', default=None,
                        help='Run only cases for this kernel.')
    parser.add_argument('--memory', default=None,
                        help='Run only cases with this host memory mode.')
    parser.add_argument('--execution', default=None,
                        help='Run only cases with this execution mode.')
    parser.add_argument('--group', default=None,
                        help='Run only cases in this group.')
    return parser.parse_args(argv)


def build_report_stem(case: ExperimentCase, build_tag: str) -> str:
    return f'{sanitize_id(case.id)}__build{sanitize_id(build_tag)}'


def filter_cases(cases: list[ExperimentCase], args: argparse.Namespace) -> list[ExperimentCase]:
    filtered = cases
    if args.case_id:
        wanted = sanitize_id(args.case_id)
        filtered = [case for case in filtered if sanitize_id(case.id) == wanted]
    if args.backend:
        filtered = [case for case in filtered if case.backend == args.backend]
    if args.kernel:
        filtered = [case for case in filtered if case.kernel == args.kernel]
    if args.memory:
        filtered = [case for case in filtered if case.memory_host == args.memory]
    if args.execution:
        filtered = [case for case in filtered if case.execution_mode == args.execution]
    if args.group:
        filtered = [case for case in filtered if case.group == args.group]

    return filtered


def main() -> int:
    args = parse_args()
    matrix_path = pathlib.Path(args.matrix)
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cli_override_args = build_cli_override_args(args)

    profile_runner = pathlib.Path(args.profile_runner) if args.profile_runner else pathlib.Path(
        __file__).with_name('profile_runner.py')
    matrix = load_experiment_matrix(matrix_path)
    experiment_name = matrix.experiment
    entries = filter_cases(matrix.cases, args)

    if not entries:
        print(f'No experiment entries matched filters in {matrix_path}')
        return 0

    for entry in entries:
        variant_dir = profile_case_dir(
            output_root,
            experiment_name,
            args.tool,
            entry.id,
        )
        report_stem = build_report_stem(entry, args.build_tag)

        command = [
            sys.executable,
            str(profile_runner),
            '--tool', args.tool,
            '--config', str(matrix_path),
            '--id', entry.id,
            '--output-dir', str(variant_dir),
            '--report-stem', report_stem,
            *cli_override_args,
        ]

        print(f'Running matrix entry {entry.id}:', ' '.join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode

    return 0


if __name__ == '__main__':
    sys.exit(main())
