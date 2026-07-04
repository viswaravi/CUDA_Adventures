# Shared Memory Bank Conflicts

## Question

Are multiple threads competing for the same bank?

## Metrics

- Shared Memory Bank Conflict metrics

## Symptoms

Shared memory throughput lower than expected.

## Typical Patterns

Bad:

tile[k][threadIdx.x]

Good:

tile[threadIdx.x][k]

## Optimization Options

- padding
- transpose layout
- warp-level primitives

## Counterexamples

Padding increases shared memory footprint.