#!/usr/bin/env python3
"""Wrap one backend experiment command with Nsight Compute or Nsight Systems.

The command can be provided directly with --binary and trailing program args,
or derived from a normalized experiment YAML via --config and --id.
"""

import argparse
import datetime
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from backend_runners import (
    BackendCommand,
    BackendRunnerError,
    build_backend_command,
    make_backend_runners,
)
from experiment_schema import load_experiment_matrix, sanitize_id
from profile_paths import profile_case_dir


DEFAULT_PROFILER_BIN = {
    'ncu': '/usr/bin/ncu',
    'nsys': '/usr/bin/nsys',
}


@dataclass(frozen=True)
class ProfileInvocation:
    binary: str
    program_args: list[str]
    report_stem: str
    output_dir: pathlib.Path
    profiler_config: dict[str, Any]


def append_option(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def _config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_profiler_options(
    args: argparse.Namespace,
    profiler_config: dict[str, Any] | None = None,
) -> list[str]:
    config = profiler_config or {}
    profiler_options: list[str] = []

    if args.tool == 'ncu':
        replay_mode = args.ncu_replay_mode or _config_value(
            config, 'replay-mode', 'replay_mode')
        profiler_set = args.ncu_profiler_set or _config_value(
            config, 'profiler-set', 'profiler_set', 'set')
        sections = args.ncu_sections or _as_list(
            _config_value(config, 'sections', 'section'))
        append_option(profiler_options, '--replay-mode', replay_mode)
        append_option(profiler_options, '--set', profiler_set)
        for section in sections:
            profiler_options.extend(['--section', section])
    else:
        trace = args.nsys_trace or _config_value(config, 'trace')
        sample = args.nsys_sample or _config_value(config, 'sample')
        capture_range = args.nsys_capture_range or _config_value(
            config, 'capture-range', 'capture_range')
        append_option(profiler_options, '--trace', trace)
        append_option(profiler_options, '--sample', sample)
        append_option(profiler_options, '--capture-range', capture_range)

    profiler_options.extend(_as_list(_config_value(config, 'args')))
    profiler_options.extend(args.profiler_args)
    return profiler_options


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Nsight profiler and store timestamped reports.'
    )
    parser.add_argument('--tool', '--profiler', choices=['ncu', 'nsys'], required=True)
    parser.add_argument('--profiler-bin', default=None)
    parser.add_argument('--binary', default=None,
                        help='Default executable for direct wrapping or selected backend')
    parser.add_argument('--cuda-binary', default=None,
                        help='CUDA backend executable; defaults to --binary')
    parser.add_argument('--triton-binary', default=None,
                        help='Triton backend runner script; defaults to --binary')
    parser.add_argument('--cublas-binary', default=None,
                        help='cuBLAS backend executable; defaults to --binary')
    parser.add_argument('--cudnn-binary', default=None,
                        help='cuDNN backend executable; defaults to --binary')
    parser.add_argument('--config', default=None,
                        help='Optional normalized experiment matrix YAML')
    parser.add_argument('--id', dest='case_id', default=None,
                        help='Experiment case id to profile when --config is used')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--report-stem', default=None)
    parser.add_argument(
        '--profiler-arg',
        dest='profiler_args',
        action='append',
        default=[],
        help='Repeatable raw profiler argument passed before the binary. Use = when the value starts with --.',
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
    parser.add_argument('program_args', nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def resolve_profiler_bin(tool: str, profiler_bin: str | None) -> str:
    if profiler_bin:
        return profiler_bin
    default_path = DEFAULT_PROFILER_BIN.get(tool)
    if default_path and pathlib.Path(default_path).exists():
        return default_path
    return tool


def split_backend_command(command: BackendCommand) -> tuple[str, list[str]]:
    argv = command.argv()
    if not argv:
        raise ValueError('Backend command is empty')
    return argv[0], argv[1:]


def _merge_profiler_config(
    matrix_config: dict[str, dict[str, Any]],
    case_config: dict[str, dict[str, Any]],
    tool: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(matrix_config.get(tool, {}))
    merged.update(case_config.get(tool, {}))
    return merged


def resolve_case_command(args: argparse.Namespace) -> ProfileInvocation:
    if not args.config:
        if not args.binary:
            raise ValueError('--binary is required when --config is not used')
        program_args = args.program_args
        if program_args and program_args[0] == '--':
            program_args = program_args[1:]
        report_stem = args.report_stem or pathlib.Path(args.binary).stem
        return ProfileInvocation(
            binary=args.binary,
            program_args=program_args,
            report_stem=report_stem,
            output_dir=pathlib.Path(args.output_dir or 'profiling'),
            profiler_config={},
        )

    if not args.case_id:
        raise ValueError('--id is required when --config is used')

    matrix = load_experiment_matrix(pathlib.Path(args.config))
    wanted = sanitize_id(args.case_id)
    matches = [case for case in matrix.cases if sanitize_id(case.id) == wanted]
    if not matches:
        raise ValueError(f"Experiment case id not found: {args.case_id}")
    if len(matches) > 1:
        raise ValueError(f"Experiment case id is not unique: {args.case_id}")

    runners = make_backend_runners(
        binary=args.binary,
        cuda_binary=args.cuda_binary,
        triton_binary=args.triton_binary,
        cublas_binary=args.cublas_binary,
        cudnn_binary=args.cudnn_binary,
    )
    backend_command = build_backend_command(matches[0], runners)
    binary, program_args = split_backend_command(backend_command)
    report_stem = args.report_stem or matches[0].id
    output_dir = pathlib.Path(args.output_dir or 'profiling')
    if args.output_dir is None:
        output_dir = profile_case_dir(
            output_dir,
            matrix.experiment,
            args.tool,
            matches[0].id,
        )
    return ProfileInvocation(
        binary=binary,
        program_args=program_args,
        report_stem=report_stem,
        output_dir=output_dir,
        profiler_config=_merge_profiler_config(
            matrix.profilers,
            matches[0].profilers,
            args.tool,
        ),
    )


def main() -> int:
    args = parse_args()
    try:
        invocation = resolve_case_command(args)
    except (BackendRunnerError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    invocation.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = invocation.output_dir / f'{invocation.report_stem}__ts{timestamp}'
    profiler_options = build_profiler_options(args, invocation.profiler_config)
    profiler_bin = resolve_profiler_bin(args.tool, args.profiler_bin)

    if args.tool == 'ncu':
        command = [
            profiler_bin,
            '--target-processes',
            'all',
            '-f',
            '-o',
            str(output_stem),
            *profiler_options,
            invocation.binary,
            *invocation.program_args,
        ]
    else:
        command = [
            profiler_bin,
            'profile',
            '--force-overwrite=true',
            '-o',
            str(output_stem),
            *profiler_options,
            invocation.binary,
            *invocation.program_args,
        ]

    print('Running:', ' '.join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == '__main__':
    sys.exit(main())
