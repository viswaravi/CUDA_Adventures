# Occupancy

## Question

Can the GPU keep enough warps resident to hide latency?

## Metrics

- Achieved Occupancy
- Theoretical Occupancy
- Active Warps per SM

## Common Causes

### Register Pressure

Symptoms:

- Registers/thread high
- Occupancy drops

Evidence:

- launch__registers_per_thread

Possible Fixes:

- reduce live ranges
- split kernels
- reduce unrolling

---

### Shared Memory Usage

Symptoms:

- Large shared allocation

Evidence:

- shared memory per block

Possible Fixes:

- smaller tiles
- recompute instead of caching

## Important Lesson

Higher occupancy is NOT always faster.

Counterexamples:

- Tensor Core kernels
- Compute-bound GEMM
- Register-heavy kernels

Always validate with measurements.