"""
Shared helpers for per-op Triton run.py launchers.

Each op's run.py creates a parser with make_arg_parser(), adds any op-specific
flags, then calls parse_triton_args() to get the final namespace.  The
export_kernel_ptx() helper writes PTX produced by triton.compile() to disk.

Usage example (in an op's run.py):

    from triton_runner_utils import make_arg_parser, export_kernel_ptx
    import my_kernels

    parser = make_arg_parser()
    args = parser.parse_args()

    if args.export_ptx:
        export_kernel_ptx(my_kernels.my_kernel, {"BLOCK_SIZE": 1024}, args.export_ptx)
"""

import argparse
import pathlib
import sys
from typing import Any, Dict, Optional


def make_arg_parser(description: str = "Triton kernel runner") -> argparse.ArgumentParser:
    """
    Return a pre-configured ArgumentParser with the standard flags shared by
    all Triton op runners.  Callers may add op-specific arguments before
    calling parser.parse_args().

    Standard flags
    --------------
    --op            Operation name (default "unknown").
    --kernel        Stable kernel name exposed by the backend runner (required).
    --dtype         Input/output dtype (default "float32").
    --n             Problem size; interpretation is op-specific (default 1 048 576).
    --device        CUDA device index (default 0).
    --warmup        Warm-up iterations before timing (default 25).
    --repeats       Timed repetitions for throughput measurement (default 100).
    --export-ptx    Optional path; if given, PTX is written here and the runner exits.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--op", default="unknown",
                        help="Operation name")
    parser.add_argument("--kernel", required=True,
                        help="Stable kernel name exposed by the backend runner")
    parser.add_argument("--dtype", default="float32",
                        help="Input/output dtype")
    parser.add_argument("--n", type=int, default=1_048_576,
                        help="Problem size (elements)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warm-up iterations")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Timed repetitions for throughput measurement")
    parser.add_argument("--export-ptx", metavar="PATH", default=None,
                        help="Write compiled PTX to this path and exit")
    return parser


def export_kernel_ptx(kernel_fn: Any, compiler_kwargs: Dict[str, Any],
                      output_path: str) -> None:
    """
    Compile *kernel_fn* with triton.compile() using *compiler_kwargs* as
    constexpr specialisation values and write the resulting PTX to
    *output_path*.

    Parameters
    ----------
    kernel_fn       : triton JIT-compiled function object.
    compiler_kwargs : dict of constexpr argument names → values used to
                      specialise the compilation (e.g. {"BLOCK_SIZE": 1024}).
    output_path     : destination file path for the PTX text.
    """
    try:
        import triton
        import triton.compiler as tc
    except ImportError:
        print("ERROR: triton is not installed. Run: pip install triton",
              file=sys.stderr)
        sys.exit(1)

    # Build the ASTSource that triton.compile() accepts
    src = tc.ASTSource(
        fn=kernel_fn,
        constants=compiler_kwargs,
        signature={},   # filled by triton from the kernel signature
    )
    compiled = triton.compile(src)
    ptx_text: str = compiled.asm.get("ptx", "")
    if not ptx_text:
        print("WARNING: triton.compile() returned no PTX for this kernel.",
              file=sys.stderr)

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ptx_text)
    print(f"PTX written to: {out}")
