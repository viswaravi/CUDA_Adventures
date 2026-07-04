## Lessons Learned

Lesson 1

Triton prefers larger tiles than my implementation.

Reason:
Improved data reuse.

Evidence:
Higher L2 hit rate.

Applies To:
GEMM
Attention

Does Not Apply:
Bandwidth-bound vector operations.



## Metrics
# MatMul Comparison Matrix

| Variant | Time | Occupancy | Registers | DRAM BW | Notes |
|----------|--------|----------|-----------|---------|--------|
| CUDA Naive | | | | | |
| CUDA Tiled | | | | | |
| CUDA Vectorized | | | | | |
| CUDA TensorCore | | | | | |
| Triton Baseline | | | | | |
| Triton Tuned | | | | | |
| Triton Autotuned | | | | | |