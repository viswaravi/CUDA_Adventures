# Dispatch Layer — Design Plan

**Status:** Deferred. Folder structure and kernel static libs are in place.
Implement once the per-op kernel library surface is stable.

---

## Goal

A thin C++ library (`libkernel_dispatch.a`) that the inference engine links
against to obtain the fastest available kernel for an op at a given shape,
dtype, and hardware target — without coupling the engine to any specific
backend.

---

## Directory Layout

```
dispatch/
  include/
    op_descriptor.hpp      # OpDescriptor, KernelHandle, HardwareInfo
    selection_strategy.hpp # abstract SelectionStrategy base
    autotune_strategy.hpp  # AutotuneStrategy : SelectionStrategy
    kernel_selector.hpp    # KernelSelector — the engine-facing API
    backend_registry.hpp   # BackendRegistry + REGISTER_BACKEND macro
  src/
    autotune_strategy.cu   # cudaEvent_t timing loop
    backend_registry.cpp   # global registry singleton
  ops/                     # per-op registration units
    reduce_backends.cu
    matmul_backends.cu
    vector_add_backends.cu
    conv2d_backends.cu
  meson.build
```

---

## Core Types

### `op_descriptor.hpp`

```cpp
struct OpDescriptor {
    std::string op;      // "matmul", "reduce", "vector_add", ...
    int         M = 0, N = 0, K = 0;
    long long   n = 0;   // for 1-D ops
    std::string dtype = "fp32";
};

struct KernelHandle {
    std::string backend_name;  // "cuda_warpshuffle", "triton_basic", ...
};

struct HardwareInfo {
    std::string arch;          // "sm_86"
    int         device_id;
    int         sm_count;
    size_t      l2_bytes;
    size_t      shared_mem_per_sm;
};

HardwareInfo hw_info_from_device(int device_id);
```

### `selection_strategy.hpp`

```cpp
struct TimingConfig {
    // Buffers pre-allocated by the autotuner for each benchmark run
    void*         d_input   = nullptr;
    void*         d_output  = nullptr;
    long long     n         = 0;
    dim3          grid, block;
    size_t        smem      = 0;
};

struct BackendEntry {
    std::string                         name;
    std::function<void(const TimingConfig&)> launch;
};

class SelectionStrategy {
public:
    virtual ~SelectionStrategy() = default;
    virtual KernelHandle select(
        const OpDescriptor&,
        const std::vector<BackendEntry>&,
        const HardwareInfo&) = 0;
};
```

### `kernel_selector.hpp`

```cpp
class KernelSelector {
public:
    explicit KernelSelector(std::unique_ptr<SelectionStrategy> strategy);

    // Look up registered backends for op.op, run the strategy, return winner.
    KernelHandle select(const OpDescriptor& op, const HardwareInfo& hw);

private:
    std::unique_ptr<SelectionStrategy> strategy_;
};
```

### `backend_registry.hpp`

```cpp
class BackendRegistry {
public:
    static BackendRegistry& instance();
    void register_backend(const std::string& op,  BackendEntry entry);
    const std::vector<BackendEntry>& backends_for(const std::string& op) const;
};

// Convenience macro — use at file scope in dispatch/ops/*.cu
#define REGISTER_BACKEND(op_name, backend_name, launch_lambda)         \
    static bool _reg_##op_name##_##backend_name = []() {               \
        BackendRegistry::instance().register_backend(                   \
            op_name, BackendEntry{backend_name, launch_lambda});        \
        return true;                                                    \
    }()
```

---

## AutotuneStrategy

```cpp
// autotune_strategy.hpp
struct AutotuneConfig {
    int warmup_iters = 5;    // -Dautotune_warmup_iters=N
    int timed_iters  = 20;   // -Dautotune_timed_iters=N
};

class AutotuneStrategy : public SelectionStrategy {
public:
    explicit AutotuneStrategy(AutotuneConfig cfg = {});

    KernelHandle select(
        const OpDescriptor&,
        const std::vector<BackendEntry>&,
        const HardwareInfo&) override;

private:
    AutotuneConfig cfg_;
    float time_backend(const BackendEntry&, const TimingConfig&);
};
```

`AutotuneStrategy::select()` algorithm:

1. Allocate `TimingConfig` buffers sized for `OpDescriptor` (shape + dtype).
2. For each `BackendEntry`:
   - Warmup: call `launch()` `cfg_.warmup_iters` times.
   - Timed: call `cudaEventRecord`, repeat `launch()` `cfg_.timed_iters` times,
     call `cudaEventRecord`, `cudaEventSynchronize`, compute elapsed_ms.
   - Store median elapsed_ms.
3. Return `KernelHandle` for the minimum-median backend.
4. Free `TimingConfig` buffers.

---

## Backend Registration (example)

```cpp
// dispatch/ops/reduce_backends.cu
#include "ops/reduce/kernels/cuda/kernels.cuh"
#include "dispatch/include/backend_registry.hpp"

REGISTER_BACKEND("reduce", "cuda_warpshuffle",
    [](const TimingConfig& cfg) {
        reduce7<<<cfg.grid, cfg.block, cfg.smem>>>(
            (float*)cfg.d_input, (float*)cfg.d_output, cfg.n);
    });

REGISTER_BACKEND("reduce", "cuda_full_unroll",
    [](const TimingConfig& cfg) {
        reduce6<<<cfg.grid, cfg.block, cfg.smem>>>(
            (float*)cfg.d_input, (float*)cfg.d_output, cfg.n);
    });
```

Adding a Triton backend later (via subprocess or Python-C++ bridge):

```cpp
REGISTER_BACKEND("reduce", "triton_basic",
    [](const TimingConfig& cfg) {
        // invoke triton launcher via popen or a pre-linked Python bridge
        run_triton_reduce_basic(cfg.d_input, cfg.d_output, cfg.n);
    });
```

---

## meson.build (dispatch/)

```meson
libkernel_dispatch = static_library(
    'kernel_dispatch',
    ['src/autotune_strategy.cu',
     'src/backend_registry.cpp',
     'ops/reduce_backends.cu',
     'ops/matmul_backends.cu',
     'ops/vector_add_backends.cu',
     'ops/conv2d_backends.cu'],
    dependencies: [
        libops_reduce_cuda_dep,
        libops_matmul_cuda_dep,
        libops_vector_add_cuda_dep,
        libops_conv2d_cuda_dep,
    ],
    cuda_args: cuda_args)

libkernel_dispatch_dep = declare_dependency(
    link_with: libkernel_dispatch,
    include_directories: include_directories('include'))
```

New `meson_options.txt` entries needed:

```meson
option('autotune_warmup_iters', type: 'integer', value: 5,
       description: 'Warm-up iterations per backend in AutotuneStrategy')
option('autotune_timed_iters',  type: 'integer', value: 20,
       description: 'Timed iterations per backend in AutotuneStrategy')
```

---

## Engine Integration (C++)

```cpp
#include "dispatch/include/kernel_selector.hpp"
#include "dispatch/include/autotune_strategy.hpp"

// At engine startup
KernelSelector selector(std::make_unique<AutotuneStrategy>());
HardwareInfo   hw = hw_info_from_device(0);

// When the engine encounters a "reduce" op in the graph
KernelHandle h = selector.select(
    {.op = "reduce", .n = 8192, .dtype = "fp32"}, hw);

// h.backend_name → "cuda_warpshuffle" (or whichever was fastest)
// The engine can cache h keyed by (op, shape-bucket, dtype, arch)
// and skip re-tuning for subsequent same-shape calls.
```

---

## Future Extensions

| Feature | Notes |
|---|---|
| **Persistent benchmark DB** | Add a `CachedStrategy` that wraps `AutotuneStrategy`, checks a JSON/SQLite file keyed by `(op, shape_bucket, dtype, arch)`, and writes winners back after tuning. |
| **Rule-based fallback** | A `HeuristicStrategy` that picks backends deterministically from shape/dtype/arch without timing — useful for latency-critical cold-start. |
| **Triton backend bridge** | Either a pybind11 module or a lightweight subprocess launcher so Triton kernels can be registered with `REGISTER_BACKEND` and timed like CUDA ones. |
| **Multi-device support** | `HardwareInfo` already carries `device_id`; `KernelSelector` can maintain a per-device cache. |
