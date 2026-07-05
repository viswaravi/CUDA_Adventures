# CUDA Adventures

## Overview
This repository contains a series of progressively advancing CUDA programming tasks designed to strengthen my understanding of GPU programming and optimization focused on Computer Vision and LLMs. Work in progress.

## Sections
### **1. Vector Addition**
**Goal:** Refresh the CUDA fundamentals and implement basic vector addition kernels. Kernel profiling using NSight Compute and NSight Systems.

#### What have I done
- Project Setup
- Basic Error Handling and Memory Cleanup
- Paged Memory vs Pinned Memory
- Chunk Processing of large data
- Multi Stream execution for large data processing speedup
- profiling and comparing using developer tools

### **2. Reduction, Prefix Sum**
**Goal:** Deepen the understanding of memory access patterns and warp-level optimizations and shared memory usage for 1D data. Implement optimized version of kernels and compare performance gains through profiling.

### **3. Matrices - Multiplication, Rotation**
**Goal:** Deepen the understanding of memory access patterns and warp-level optimizations and shared memory usage for 2D data. Implement kernels for matrix multiplication, rotation.

### **4. Computer Vision - Convolution, Fast Convolution Kernels**
**Goal:** Optimize the image processing kernels for key operations in Convolutional Neural Networks (CNNs), Fast Convolution. 

<!-- Integration with CUDA Graphs and benchmark the performance difference. -->
<!-- ### **5. Fast Matrix Multiplication**
**Goal:** Optimize matrix multiplication using hardware-specific features. Utilize Tensor Cores. Test performance scaling.

### **6. Transformer Attention Optimization**
**Goal:** Develop and optimize kernels for softmax and self-attention mechanisms used in Transformer architectures.

### **7. Sparse Matrix Operations for LLM Efficiency**
**Goal:** Optimize sparse matrix operations, crucial for large language model (LLM) efficiency.

### **8. Model Quantization & TensorRT Optimization**
**Goal:** Optimize models for inference by applying quantization and leveraging TensorRT.

### **9. Optimizing GPT Model**
**Goal:** Apply all learned CUDA techniques to optimize a GPT-style model for inference and training efficiency. Transformer layers and custom KV-cache implementation.

### **10. Multi-GPU Training & NCCL Optimization (Project 8)**
**Goal:** Scale the  optimizations across multiple GPUs for training large models. -->

---

## Compiling

All tasks share a common build system based on Meson. Each task directory contains a `meson.build` file defining the build targets for that task. Configure once, then build the target you are actively working on:
```bash
meson setup build

# Compile a specific target
meson compile -C build <target-name>
```

CUDA build flags are owned by the root Meson file and shared by every op.
`enable_cuda_fast_math` and `enable_cuda_lineinfo` default to `true`. Disable
them when you need strict math behavior or builds without source line mapping:

```bash
meson setup build -Dgpu_arch=sm_86 -Denable_cuda_fast_math=false -Denable_cuda_lineinfo=false
```

Use native Meson build types for profiling or release-style builds:

```bash
meson setup build --buildtype=release -Dgpu_arch=sm_86 -Denable_cuda_lineinfo=true
```

CUDA ops use backend-owned runners with normalized flags. To build a single backend runner target:

```bash
meson compile -C build ops_vector_add_cuda_runner
```

---

## Profiling

Reports are stored under `profiling/<experiment>/<tool>/<id parts>/`, where the
normalized sweep id is split on underscores. For example,
`cuda_pinned_scalar_64m` writes under
`profiling/vector_addition/ncu/cuda/pinned/scalar/64m/`.

Backend runner executables are declared in each operation's `experiments.yaml`
under `backend_capabilities.<backend>.executable`. CLI flags such as `--binary`
or `--cuda-binary` override the YAML path for ad hoc runs.

### Prerequisites

Activate the project virtual environment once before running any profiling commands:

```bash
source .venv/bin/activate
```


---

### Single-shot profiling with custom args (`profile_runner.py`)

Thin wrapper — pass explicit args, get a timestamped report. No YAML awareness.

**Syntax**
```bash
python tools/profile_runner.py \
  --tool <ncu|nsys> \
  --profiler-bin <path-to-profiler> \
  --binary <path-to-binary> \
  --output-dir <output-directory> \
  --report-stem <report-name> \
  [--ncu-replay-mode <mode>] \
  [--ncu-profiler-set <set>] \
  [--ncu-section <section>]... \
  [--nsys-trace <domains>] \
  [--nsys-sample <mode>] \
  [--nsys-capture-range <range>] \
  [--profiler-arg=<raw-profiler-flag>]... \
  -- <binary-args...>
```

Use the tool-specific switches for common options such as NCU `--replay-mode` / `--set`
or NSYS `--trace` / `--sample`. With `--config` and `--id`, defaults are read
from the YAML `profilers:` block and CLI switches override them. For anything
else, repeat `--profiler-arg=<flag>`.

**Example — NCU, CUDA vector-add**
```bash
python tools/profile_runner.py \
  --tool ncu \
  --profiler-bin /usr/bin/ncu \
  --binary ./build/ops/vector_add/ops_vector_add_cuda_runner \
  --output-dir profiling/vector_addition/ncu/cuda/manual \
  --report-stem cuda_pageable_scalar \
  --ncu-replay-mode kernel \
  --ncu-profiler-set full \
  -- --op vector_add --kernel scalar --dtype int32 \
     --n 67108864 --memory-host pageable --execution-mode single_stream
```

**Example — NSYS from normalized config**
```bash
python tools/profile_runner.py \
  --config ops/vector_add/experiments.yaml \
  --id cuda_pinned_s4_vectorized_512m \
  --tool nsys \
  --profiler-bin /usr/bin/nsys
```

**Example — NCU from one normalized sweep**
```bash
python tools/profile_runner.py \
  --config ops/vector_add/experiments.yaml \
  --id cuda_pinned_scalar_64m \
  --tool ncu \
  --profiler-bin /usr/bin/ncu
```

---

### Full experiment matrix sweep (`matrix_runner.py`)

Runs all sweep entries defined in a YAML experiment matrix in sequence.
Output paths are derived automatically from the matrix metadata — no manual path construction needed.

**Syntax**
```bash
python tools/matrix_runner.py \
  --matrix <path-to-yaml> \
  --tool <ncu|nsys> \
  [--binary <path-to-binary>] \
  [--cuda-binary <path>] \
  [--triton-binary <path>] \
  [--cublas-binary <path>] \
  [--profiler-bin <path>]   # defaults: ncu=/usr/bin/ncu, nsys=/usr/bin/nsys \
  [--output-dir <dir>]      # default: profiling \
  [--build-tag <tag>]       # default: dev \
  [--ncu-replay-mode <mode>] \
  [--ncu-profiler-set <set>] \
  [--ncu-section <section>]... \
  [--nsys-trace <domains>] \
  [--nsys-sample <mode>] \
  [--nsys-capture-range <range>] \
  [--profiler-arg=<raw-profiler-flag>]... \
  [--id <case-id>]          # run only this normalized case \
  [--backend <backend>] \
  [--kernel <kernel>] \
  [--memory <host-memory>] \
  [--execution <mode>] \
  [--group <group>]
```

The YAML file can also define profiler defaults:

```yaml
experiment: vector_addition
profilers:
  ncu:
    replay-mode: kernel
    profiler-set: full
    sections: []
    args: []
  nsys:
    trace: cuda,nvtx,osrt
    sample: none
    args: []
```

Each sweep entry may optionally add its own `profilers:` block. Matrix defaults are applied first,
then per-entry overrides, then explicit CLI flags.

**Example — NCU sweep over all vec_add experiments**
```bash
python tools/matrix_runner.py \
  --matrix ops/vector_add/experiments.yaml \
  --tool ncu \
  --ncu-replay-mode kernel \
  --ncu-profiler-set full
```

**Example — NSYS sweep with a custom build tag**
```bash
python tools/matrix_runner.py \
  --matrix ops/vector_add/experiments.yaml \
  --tool nsys \
  --build-tag v1
```

**Example — NCU, single normalized case**

Use `--id` to pick one entry from the matrix without writing an ad-hoc YAML file.

```bash
# CUDA stream-overlap case only
python tools/matrix_runner.py \
  --matrix ops/vector_add/experiments.yaml \
  --tool ncu \
  --id cuda_pinned_s4_vectorized_512m
```

```bash
# Triton cases only
python tools/matrix_runner.py \
  --matrix ops/vector_add/experiments.yaml \
  --tool ncu \
  --backend triton
```

Reports land at `profiling/<experiment>/<tool>/<id parts>/`, where `<id parts>`
comes from splitting the normalized case id on underscores.

---
