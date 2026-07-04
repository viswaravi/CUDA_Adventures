# Profiling based Optimization Checklist

## Step 1

Classify Kernel

- Memory bound?
- Compute bound?
- Latency bound?

Evidence:

__________________

## Step 2

Occupancy

Theoretical:

__________________

Achieved:

__________________

Limiting Resource:

__________________

## Step 3

Memory

Global Load Efficiency:

__________________

Global Store Efficiency:

__________________

DRAM Utilization:

__________________

## Step 4

Execution

Warp Stall Reasons:

__________________

Top Stall:

__________________

## Step 5

Optimization Candidate

Chosen Optimization:

__________________

Reason:

__________________

## Step 6

Result

Speedup:

__________________

Why:

__________________

## Step 7

Generalized Rule

When should I apply this again?

__________________

When should I NOT apply it?

__________________


# Triton vs CUDA Comparison Checklist
Comparison should be done at the level of:

```bash
Algorithm
    ↓
Kernel Design
    ↓
Compiler Output
    ↓
Hardware Behavior
```

## Algorithm
- Tiling strategy
- Reduction strategy
- Data layout

## Kernel Design
- Threads/block
- Warps/block
- Shared memory
- Vectorization
- Divergence
- Synchronization strategy

## Compiler Output
- Instruction count
- Register count
- Memory instructions
- Control flow instructions
- Tensor Core instructions


## Hardware Behavior
- Occupancy
- Warp stalls
- L2 hits
- Bandwidth
- IPC