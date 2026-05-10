#!/usr/bin/env python3

import argparse
import datetime
import pathlib
import subprocess
import sys


def append_option(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def build_profiler_options(args: argparse.Namespace) -> list[str]:
    profiler_options: list[str] = []

    if args.tool == 'ncu':
        append_option(profiler_options, '--replay-mode', args.ncu_replay_mode)
        append_option(profiler_options, '--set', args.ncu_profiler_set)
        for section in args.ncu_sections:
            profiler_options.extend(['--section', section])
    else:
        append_option(profiler_options, '--trace', args.nsys_trace)
        append_option(profiler_options, '--sample', args.nsys_sample)
        append_option(profiler_options, '--capture-range',
                      args.nsys_capture_range)

    profiler_options.extend(args.profiler_args)
    return profiler_options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Nsight profiler and store timestamped reports.'
    )
    parser.add_argument('--tool', choices=['ncu', 'nsys'], required=True)
    parser.add_argument('--profiler-bin', required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--report-stem', required=True)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    program_args = args.program_args
    if program_args and program_args[0] == '--':
        program_args = program_args[1:]

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f'{args.report_stem}__ts{timestamp}'
    profiler_options = build_profiler_options(args)

    if args.tool == 'ncu':
        command = [
            args.profiler_bin,
            '--target-processes',
            'all',
            '-f',
            '-o',
            str(output_stem),
            *profiler_options,
            args.binary,
            *program_args,
        ]
    else:
        command = [
            args.profiler_bin,
            'profile',
            '--force-overwrite=true',
            '-o',
            str(output_stem),
            *profiler_options,
            args.binary,
            *program_args,
        ]

    print('Running:', ' '.join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == '__main__':
    sys.exit(main())
