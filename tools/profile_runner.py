#!/usr/bin/env python3

import argparse
import datetime
import pathlib
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Nsight profiler and store timestamped reports.'
    )
    parser.add_argument('--tool', choices=['ncu', 'nsys'], required=True)
    parser.add_argument('--profiler-bin', required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--report-stem', required=True)
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

    if args.tool == 'ncu':
        command = [
            args.profiler_bin,
            '--target-processes',
            'all',
            '-f',
            '-o',
            str(output_stem),
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
            args.binary,
            *program_args,
        ]

    print('Running:', ' '.join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == '__main__':
    sys.exit(main())
