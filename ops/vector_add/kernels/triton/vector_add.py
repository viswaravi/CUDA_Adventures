"""
Triton kernel implementations for the vector_add op.

Variants
--------
triton-basic       : Blocked load / elementwise add / store.  One element per
                     program instance, BLOCK_SIZE elements per block.
triton-vectorized  : Identical arithmetic, but uses a larger default BLOCK_SIZE
                     (4 096) and marks loads with evict_first to hint that input
                     data need not be kept in L2 after the single read.
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


# ─── Variant registry ────────────────────────────────────────────────────────

#: Maps variant name → (kernel_fn, default_BLOCK_SIZE)
VARIANTS = {
    "triton-basic":       (add_kernel,            1024),
    "triton-vectorized":  (add_kernel_vectorized,  4096),
}


def run(variant: str, n: int, device: int) -> float:
    """
    Launch *variant* on *n* float32 elements.  Returns measured throughput
    in GB/s.  Raises ValueError for unknown variant names.
    """
    import torch
    import triton.testing

    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Available: {list(VARIANTS)}"
        )

    kernel_fn, block_size = VARIANTS[variant]

    x = torch.rand(n, dtype=torch.float32, device=f"cuda:{device}")
    y = torch.rand(n, dtype=torch.float32, device=f"cuda:{device}")
    out = torch.empty_like(x)

    def grid(meta): return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # Correctness check
    kernel_fn[grid](x, y, out, n, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    expected = x + y
    if not torch.allclose(out, expected, atol=1e-5):
        raise RuntimeError(f"Correctness check FAILED for variant {variant!r}")
    print(f"[{variant}] Correctness: PASS")

    # Throughput benchmark
    ms = triton.testing.do_bench(
        lambda: kernel_fn[grid](x, y, out, n, BLOCK_SIZE=block_size)
    )
    gbps = (3 * x.numel() * x.element_size()) / ms * 1e-6  # 2 reads + 1 write
    print(f"[{variant}] n={n:,}  time={ms:.3f} ms  throughput={gbps:.1f} GB/s")
    return gbps


def export_ptx(variant: str, output_path: str, block_size: int | None = None) -> None:
    """
    AOT-compile *variant* with triton.compile() and write PTX to *output_path*.
    """
    import sys
    try:
        import triton
        import triton.compiler as tc
    except ImportError:
        print("ERROR: triton is not installed.", file=sys.stderr)
        sys.exit(1)

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")

    kernel_fn, default_bs = VARIANTS[variant]
    bs = block_size if block_size is not None else default_bs

    src = tc.ASTSource(fn=kernel_fn, constants={
                       "BLOCK_SIZE": bs}, signature={})
    compiled = triton.compile(src)
    ptx_text = compiled.asm.get("ptx", "")

    import pathlib
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ptx_text)
    print(f"PTX written to: {out}")
