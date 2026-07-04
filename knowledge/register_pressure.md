# Register Pressure

## Question

Are too many registers limiting occupancy?

## Metrics

- Registers per thread
- Spills
- Local memory traffic

## Evidence

Look for:

- local memory loads
- local memory stores

## Common Causes

- aggressive inlining
- excessive unrolling
- large temporary arrays
- vectorization

## Optimization Options

- reduce unroll factor
- shorten live ranges
- split kernel

## Counterexamples

Reducing registers may reduce instruction-level parallelism.