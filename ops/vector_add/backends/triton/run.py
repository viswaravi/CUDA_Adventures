#!/usr/bin/env python3
"""
Triton launcher for the vector_add op.

Used directly by profile_matrix_runner.py as a drop-in replacement for the
CUDA benchmark binary. Accepts normalized experiment fields:

    profile_matrix_runner.py \\
        --matrix  ops/vector_add/experiments.yaml \\
        --triton-binary ops/vector_add/backends/triton/run.py \\
        --tool    ncu
"""

import argparse
import pathlib
import sys

# Allow importing vector_add.py from the same directory regardless of cwd
_HERE = pathlib.Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Allow importing triton_runner_utils from tools/
_TOOLS = (_HERE / "../../../../tools").resolve()
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import vector_add as _va  # noqa: E402  (import after sys.path setup)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triton vector_add runner"
    )
    parser.add_argument("--op", default="vector_add")
    parser.add_argument("--kernel", required=True,
                        help=f"Kernel name. Available: {list(_va.KERNELS)}")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--n", type=int, default=268_435_456,
                        help="Number of elements (default: 268 435 456 = 256 M)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--validate", default="true")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warm-up iterations before timing")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Timed repetitions for throughput measurement")
    parser.add_argument("--export-ptx", metavar="PATH", default=None,
                        help="Write compiled PTX to PATH and exit")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.op != "vector_add":
        print(f"ERROR: unsupported --op {args.op!r}", file=sys.stderr)
        return 2

    # PTX export mode — compile and exit without running
    if args.export_ptx:
        _va.export_ptx(args.kernel, args.export_ptx, block_size=args.block_size)
        return 0

    # Verify triton + torch are available
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
    except ImportError as exc:
        print(
            f"ERROR: {exc}\nInstall with: pip install torch triton", file=sys.stderr)
        return 1

    torch = __import__("torch")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    print(f"Device {args.device}: {props.name}  "
          f"SM {props.major}.{props.minor}  "
          f"{props.total_memory // (1024**3)} GB")

    try:
        _va.run(
            kernel=args.kernel,
            dtype=args.dtype,
            n=args.n,
            device=args.device,
            block_size=args.block_size,
            validate=args.validate.lower() in ("1", "true", "yes", "on"),
            warmup=args.warmup,
            repeats=args.repeats,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
