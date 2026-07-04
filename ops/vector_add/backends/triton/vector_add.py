"""
Triton kernel implementations for the vector_add op.

Kernels
-------
block: Blocked load / elementwise add / store.
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Elementwise add: out = x + y, blocked over BLOCK_SIZE elements."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def add_kernel_vectorized(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorised elementwise add: same arithmetic as add_kernel but uses
    evict_first eviction policy to reduce L2 pressure on large inputs.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_first")
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy="evict_first")
    tl.store(out_ptr + offsets, x + y, mask=mask)


KERNELS = {
    "block": add_kernel,
}


def run(
    *,
    kernel: str,
    dtype: str,
    n: int,
    device: int,
    block_size: int,
    validate: bool,
    warmup: int,
    repeats: int,
) -> float:
    """
    Launch vector_add on *n* elements. Returns measured throughput in GB/s.
    """
    import torch
    import triton.testing

    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Available: {list(KERNELS)}")
    if dtype != "float32":
        raise ValueError("Triton vector_add currently supports dtype='float32' only")

    kernel_fn = KERNELS[kernel]

    x = torch.rand(n, dtype=torch.float32, device=f"cuda:{device}")
    y = torch.rand(n, dtype=torch.float32, device=f"cuda:{device}")
    out = torch.empty_like(x)

    def grid(meta): return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    if validate:
        kernel_fn[grid](x, y, out, n, BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        expected = x + y
        if not torch.allclose(out, expected, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"Correctness check FAILED for kernel {kernel!r}")
        print(f"[{kernel}] Correctness: PASS")

    # Throughput benchmark
    ms = triton.testing.do_bench(
        lambda: kernel_fn[grid](x, y, out, n, BLOCK_SIZE=block_size),
        warmup=warmup,
        rep=repeats,
    )
    gbps = (3 * x.numel() * x.element_size()) / ms * 1e-6  # 2 reads + 1 write
    print(f"[{kernel}] dtype={dtype} n={n:,} block_size={block_size} "
          f"time={ms:.3f} ms  throughput={gbps:.1f} GB/s")
    return gbps


def export_ptx(kernel: str, output_path: str, block_size: int | None = None) -> None:
    """
    AOT-compile *kernel* with triton.compile() and write PTX to *output_path*.
    """
    import sys
    try:
        import triton
        import triton.compiler as tc
    except ImportError:
        print("ERROR: triton is not installed.", file=sys.stderr)
        sys.exit(1)

    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}")

    kernel_fn = KERNELS[kernel]
    bs = block_size if block_size is not None else 1024

    src = tc.ASTSource(fn=kernel_fn, constants={
                       "BLOCK_SIZE": bs}, signature={})
    compiled = triton.compile(src)
    ptx_text = compiled.asm.get("ptx", "")

    import pathlib
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ptx_text)
    print(f"PTX written to: {out}")
