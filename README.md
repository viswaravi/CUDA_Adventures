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

All tasks share a common build system based on Meson. Each task directory contains a `meson.build` file defining the build targets for that task. To compile the binaries for a task, run:
```bash
meson setup build

# Compile all targets
meson compile -C build

# Or compile a specific target
meson compile -C build <target-name>
```

---

## Profiling

Reports are stored under `profiling/<experiment>/<tool>/<variant>/<label>/`.

### Prerequisites

Activate the project virtual environment once before running any profiling commands:

```bash
source activate
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
  -- <binary-args...>
```

**Example — NCU, `pageable/small`**
```bash
python tools/profile_runner.py \
  --tool ncu \
  --profiler-bin /usr/bin/ncu \
  --binary ./build/1_vector_addition/vec_add \
  --output-dir profiling/vec_add/ncu/pageable/small \
  --report-stem pageable_small \
  -- --variant pageable --n 67108864
```

**Example — NSYS, `pinned/medium`**
```bash
python tools/profile_runner.py \
  --tool nsys \
  --profiler-bin /usr/bin/nsys \
  --binary ./build/1_vector_addition/vec_add \
  --output-dir profiling/vec_add/nsys/pinned/medium \
  --report-stem pinned_medium \
  -- --variant pinned --n 268435456
```

---

### Full experiment matrix sweep (`profile_matrix_runner.py`)

Runs all sweep entries defined in a YAML experiment matrix in sequence.
Output paths are derived automatically from the matrix metadata — no manual path construction needed.

**Syntax**
```bash
python tools/profile_matrix_runner.py \
  --matrix <path-to-yaml> \
  --tool <ncu|nsys> \
  --binary <path-to-binary> \
  [--profiler-bin <path>]   # defaults: ncu=/usr/bin/ncu, nsys=/usr/bin/nsys \
  [--output-dir <dir>]      # default: profiling \
  [--build-tag <tag>]       # default: dev \
  [--label <label>]         # run only this entry (single-shot mode) \
  [--variant <variant>]     # disambiguate when multiple entries share the same label
```

**Example — NCU sweep over all vec_add experiments**
```bash
python tools/profile_matrix_runner.py \
  --matrix 1_vector_addition/vec_add_experiments.yaml \
  --tool ncu \
  --binary ./build/1_vector_addition/vec_add
```

**Example — NSYS sweep with a custom build tag**
```bash
python tools/profile_matrix_runner.py \
  --matrix 1_vector_addition/vec_add_experiments.yaml \
  --tool nsys \
  --binary ./build/1_vector_addition/vec_add \
  --build-tag v1
```

**Example — NCU, single entry by label + variant**

Use `--label` to pick one entry from the matrix without writing an ad-hoc YAML file.
If two entries have the same label (e.g. both `pageable` and `pinned` have `label: small`), add `--variant` to disambiguate.

```bash
# pageable/small only
python tools/profile_matrix_runner.py \
  --matrix 1_vector_addition/vec_add_experiments.yaml \
  --tool ncu \
  --binary ./build/1_vector_addition/vec_add \
  --variant pageable \
  --label small
```

```bash
# streamed-large/s4-256m only
python tools/profile_matrix_runner.py \
  --matrix 1_vector_addition/vec_add_experiments.yaml \
  --tool ncu \
  --binary ./build/1_vector_addition/vec_add \
  --label s4-256m
```

Reports land at `profiling/vec_add/ncu/<variant>/<label>/`.

---