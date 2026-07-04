# Global Memory Coalescing

## Question

How many memory transactions does a warp generate?

## Metrics

- Global Load Efficiency
- Global Store Efficiency
- DRAM Throughput

## Symptoms

Low bandwidth utilization.

## Evidence

Warp accesses:

thread0 -> A[0]
thread1 -> A[17]
thread2 -> A[34]

Instead of:

thread0 -> A[0]
thread1 -> A[1]
thread2 -> A[2]

## Optimization Options

- data layout changes
- structure of arrays
- vectorized loads

## Counterexamples

Vectorized loads may increase:

- register pressure
- alignment requirements