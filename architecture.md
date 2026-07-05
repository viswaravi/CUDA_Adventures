# Architecture

## Repository Model

This repository is a kernel lab organized by operation. An operation is a benchmarkable unit such as `vector_add`, `reduce`, `prefix_sum`, `matmul`, `rotation`, or `conv2d`. Each operation lives under `ops/<op>/` and owns its experiment configuration, backend implementations, Meson targets, and optional analysis notes.

The target layout for an operation is:

```text
ops/<op>/
  <op-or-experiment>.yaml
  meson.build
  include/
    kernel_lab/
      ops/
        <op>/
          <public op headers>.hpp
  backends/
    cuda/
      <op>_cuda_runner.cu
      kernels.cu
      kernels.cuh
    triton/
      run.py
      <op>.py
    cublas/
      <op>_cublas_main.cpp
    cudnn/
      <op>_cudnn_main.cpp
  docs/
```

The active CUDA operations now use this backend-owned layout. `attention`
still contains placeholder kernel directories and should follow the same layout
when it is implemented.

Shared utilities follow the same module-owned public/private split:

```text
utils/
  include/kernel_lab/utils/
    <public utility headers>.hpp
    <public utility headers>.cuh
  src/
    <utility implementation sources>.cu
```

Public C++/CUDA APIs live under the `kernel_lab` namespace. Operation public
APIs use `kernel_lab::ops::<op>`. Public headers are included by package path,
for example `<kernel_lab/utils/cuda_utils.cuh>` or
`<kernel_lab/ops/vector_add/config.hpp>`. Backend-local headers such as
`kernels.cuh` remain private quoted includes from the owning backend directory.
Vendor headers such as STB are private implementation details and are not
exported through public module dependencies.

## Experiment Schema

Experiment YAML uses normalized fields instead of overloaded variant names. A sweep case describes what is being run, not how a legacy launcher happens to name it:

```yaml
id: cuda_pinned_s4_vectorized_512m
op: vector_add
backend: cuda
kernel: vectorized
dtype: int32
memory: {host: pinned}
execution: {mode: chunked_multistream, streams: 4}
shape: {n: 536870912}
tuning: {vector_width: 4}
params: {}
```

Common top-level defaults include `op`, `dtype`, `validate`, `warmup`, and
`repeats`. Backend capability blocks declare supported kernels, dtypes, memory
modes, and execution modes. These capabilities must come from the operation
YAML; `tools/experiment_schema.py` does not carry hardcoded per-op/backend
capability defaults. The schema loader validates declared capabilities and
rejects legacy `variant:` sweeps.

Each backend capability block should also declare the backend runner executable
used by default, for example:

```yaml
backend_capabilities:
  cuda:
    executable: ./build/ops/vector_add/ops_vector_add_cuda_runner
```

## Runner Responsibilities

Python orchestration is split into three layers:

- `tools/experiment_schema.py`: load YAML into `ExperimentCase` objects and validate capabilities.
- `tools/matrix_runner.py`: expand a matrix, apply filters such as `--id`, `--backend`, `--kernel`, `--memory`, `--execution`, and `--group`, then run each case through `profile_runner.py`.
- `tools/profile_runner.py`: wrap one backend command with Nsight Compute or
  Nsight Systems. With `--config` and `--id`, it selects one sweep from the
  operation YAML, builds that backend command, applies YAML profiler defaults
  from `profilers:`, and writes the report under
  `profiling/<experiment>/<tool>/<id parts>/`, where the normalized case id is
  split on underscores.

`tools/profile_matrix_runner.py` is only a compatibility wrapper around `tools/matrix_runner.py`.

## Backend Runner Contract

`tools/backend_runners/` contains thin command builders for backend families: CUDA, Triton, cuBLAS, and cuDNN. These modules do not parse YAML, run profilers, or implement kernels. They convert an `ExperimentCase` into the argv expected by the backend-owned executable.
They do not hardcode per-op executable paths; those paths come from the
operation YAML and can still be overridden by CLI flags such as `--binary` or
`--cuda-binary`.

Backend executables should expose normalized CLI fields where applicable:

```text
--op <op>
--kernel <kernel>
--dtype <dtype>
--validate true|false
--warmup <count>
--repeats <count>
```

Shape, memory, execution, tuning, and params fields become explicit CLI flags for that backend, for example `--n`, `--memory-host`, `--execution-mode`, `--streams`, `--free-mem-threshold`, `--vector-width`, `--block-size`, or `--alpha`.

For C++/CUDA/CUDA-library backends, put shared operation config in the
operation public include tree when multiple backends consume the same fields.
For example, `ops/vector_add/include/kernel_lab/ops/vector_add/config.hpp`
defines `kernel_lab::ops::vector_add::VectorAddConfig`, the vector-add
ArgSpecs, and conversion from `kernel_lab::RunConfig`. CUDA and cuBLAS both
parse the same normalized CLI into that config, then validate only the subset
they implement.

Shared experiment-level C++ types belong in the utils public include tree. For
example, `utils/include/kernel_lab/utils/experiment_types.hpp` owns common
`Status`, `DType`, tolerance, dtype parsing, and default vector-width helpers.
`utils/include/kernel_lab/utils/cuda_dtype.cuh` owns CUDA half and bfloat16
conversion helpers for future typed runners.
`utils/include/kernel_lab/utils/host_algorithms.hpp` owns small host reference
algorithms such as CPU sum reduction that can serve multiple ops.
Operation-specific execution code should stay under the owning backend folder.

## Backend Implementation Guidance

Backend runners parse normalized CLI flags directly and dispatch through
operation-owned config structs. The old shared C++ `VariantRegistry` model has
been removed; do not add new `--variant` or `--list-variants` interfaces.
For example, vector-add CUDA dispatches on `--kernel`, `--dtype`, memory, and
execution flags.

Triton backends should be Python-native launchers that accept the same normalized fields. A Python dictionary of stable kernel names is sufficient.

Library backends such as cuBLAS or cuDNN should live under the operation backend folder and expose the same normalized operation contract. cuDNN should only be used for operations where it is meaningful, such as convolution, pooling, activation, or batchnorm.

## Vector-Add Example

Current vector-add layout:

```text
ops/vector_add/
  experiments.yaml
  include/
    kernel_lab/
      ops/
        vector_add/
          config.hpp
  backends/
    cuda/
      vector_add_cuda_runner.cu
      kernels.cu
      kernels.cuh
      thrust.cu
    triton/
      run.py
      vector_add.py
    cublas/
      vector_add_cublas_main.cpp
```

CUDA vector-add uses `vector_add_cuda_runner.cu` as the executable source. It
parses the normalized CLI through the op public config and dispatches directly
to explicit dtype-specific kernels in `backends/cuda/kernels.cu`. The CUDA
backend supports `int32`, `float32`, `float16`, and `bfloat16` with
`--kernel scalar` and `--kernel vectorized`. The vectorized CUDA path maps to
`vec4` kernels for 32-bit types and pair/half2 kernels for 16-bit types.
Triton vector-add currently supports `float32` with `--kernel block`. cuBLAS
vector-add is a `float32` SAXPY reference.

Reduce follows the same normalized CUDA runner pattern. Its public config lives
in `ops/reduce/include/kernel_lab/ops/reduce/config.hpp`, the executable source
is `ops/reduce/backends/cuda/reduce_cuda_runner.cu`, and the runner accepts
`--op reduce`, `--kernel`, `--dtype`, `--n`, `--validate`, `--warmup`, and
`--repeats`. Reduce is currently `float32`-only and dispatches every CUDA kernel
declared in `ops/reduce/experiments.yaml`, including a full-array
`warp-primitives` reduction.

Prefix-sum also uses the normalized CUDA runner pattern. Its public config lives
in `ops/prefix_sum/include/kernel_lab/ops/prefix_sum/config.hpp`, the executable
source is `ops/prefix_sum/backends/cuda/prefix_sum_cuda_runner.cu`, and the
runner accepts `--op prefix_sum`, `--kernel`, `--dtype`, `--n`, `--validate`,
`--warmup`, and `--repeats`. Prefix-sum is currently `float32`-only and
dispatches every CUDA kernel declared in `ops/prefix_sum/experiments.yaml`.

## Build and Verification

Build a migrated CUDA runner target:

```bash
meson compile -C build ops_vector_add_cuda_runner
meson compile -C build ops_reduce_cuda_runner
meson compile -C build ops_prefix_sum_cuda_runner
```

CUDA compile flags are owned by the root Meson file and shared by every active
operation. `gpu_arch` selects the target architecture, `enable_cuda_fast_math`
controls NVCC `--use_fast_math`, and `enable_cuda_lineinfo` controls NVCC
`-lineinfo` for runner targets and generated PTX/CUBIN exports. Both CUDA flag
toggles default to `true`; use native Meson build types such as
`--buildtype=release` for profiling or release-style builds.

The old shared C++ variant registry has been removed. Full-repo builds should
use normalized direct-parsing runners.

Smoke-check command generation without launching Nsight:

```bash
.venv/bin/python tools/matrix_runner.py \
  --matrix ops/vector_add/experiments.yaml \
  --id cuda_pinned_s4_vectorized_512m \
  --tool nsys \
  --profiler-bin /bin/echo
```

Run focused runner tests:

```bash
.venv/bin/python -m unittest tests/test_experiment_runners.py
```

## Maintenance Rules

When changing experiment schema, backend layout, runner behavior, build targets, or backend launcher CLI contracts, update this file in the same change. Keep `AGENTS.md` and README command examples aligned with this architecture document.
