# Vector Addition

## Goal

Understand memory subsystem behavior for a bandwidth-bound kernel.

## Kernel Characteristics

Arithmetic Intensity:
- Extremely low

Expected Bottleneck:
- Global memory bandwidth

Expected Occupancy Requirement:
- Moderate

Expected Tensor Core Usage:
- None

## Hypothesis

Pinned memory should improve H2D and D2H transfer throughput.

Increasing streams should overlap:

- H2D copy
- Kernel execution
- D2H copy

## Profiling Results

...

## Observations

Observation 1:
Pinned memory improved transfer throughput by 2.1x.

Evidence:
- memcpy throughput metric

Observation 2:
4 streams improved throughput by 35%.

Evidence:
- Nsight Systems timeline

## Generalized Rule

For transfer-dominated workloads:

- Use pinned memory
- Overlap transfers with execution
- Use multiple streams when transfers dominate

## Counterexamples

Does not help:

- Compute-bound kernels
- Small transfers
- Unified memory workloads