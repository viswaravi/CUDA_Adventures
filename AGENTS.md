# Repository Guidelines

## Project Structure & Module Organization

This repository is a CUDA kernel lab organized by operation. Shared CUDA/C++ helpers live in `utils/`, operation-specific code lives under `ops/<op>/`, and Python profiling or experiment utilities live in `tools/`. Read `architecture.md` before changing experiment schemas, backend layout, runner behavior, or build targets.

Experiments use the normalized backend-owned layout: `ops/<op>/experiments.yaml` for the experiment matrix, `ops/<op>/include/kernel_lab/ops/<op>/` for public op-level parser/config shared across backends, and `ops/<op>/backends/<backend>/` for backend runners and kernels. Generated profiler output belongs under `profiling/`, and build output belongs under `build/`.

## Build, Test, and Development Commands

Configure the Meson build:

```bash
meson setup build -Dgpu_arch=sm_75
```

Use the architecture matching your GPU, for example `sm_86`. Build the
normalized vector-add CUDA runner target:

```bash
meson compile -C build ops_vector_add_cuda_runner
```

The old shared variant registry has been removed from `utils/include/kernel_lab/utils/cli_args.hpp`.
Backend runners should parse normalized backend flags directly.
For migrated ops, compile the touched benchmark target directly:

```bash
meson compile -C build <target-name>
```

Run a benchmark binary directly:

```bash
./build/ops/vector_add/ops_vector_add_cuda_runner --op vector_add --kernel scalar --dtype int32 --n 67108864 --memory-host pageable --execution-mode single_stream
```

Run profiling sweeps from YAML matrices:

```bash
source .venv/bin/activate
python tools/matrix_runner.py --matrix ops/vector_add/experiments.yaml --tool ncu
```

## Coding Style & Naming Conventions

C++ and CUDA use C++17 and the repository `.clang-format`: LLVM base, 4-space indentation, right-aligned pointers, 120-column limit, and inserted braces. For backend-owned ops, keep kernels and runner executable sources together under `ops/<op>/backends/<backend>/`, and put shared public op config in `ops/<op>/include/kernel_lab/ops/<op>/`. Use runner target names like `ops_<op>_<backend>_runner` and Meson variables like `libops_<op>_cuda`.

Python tools use type hints, dataclasses where useful, and clear snake_case names. Keep experiment IDs and YAML keys stable because profiler output paths depend on them.

Always add docstrings to new functions and classes, and update the operation README when adding new kernels or profiling tools. Use `# TODO` comments for future work, and `# NOTE` comments for clarifications.

## Testing Guidelines

There is no standalone test suite yet. Treat each benchmark runner as the validation path: build the touched target, run at least one small input, and keep `--validate` enabled when the runner supports it. For profiling-tool changes, run a single-label matrix entry before launching full sweeps.

## Commit & Pull Request Guidelines

Recent history uses short imperative or conventional-style subjects, for example `refactor: reorganize into ops/ kernel lab structure` and `add profile matrix runner based on yaml file`. Keep commits focused by operation or tool. Pull requests should include the affected op/tool, build command used, validation command or profiler sweep run, GPU architecture, and any relevant profiling result path.

## Agent-Specific Instructions
- Always read `architecture.md` before making architectural, runner, schema, backend-layout, or build changes.
- Always update `architecture.md` in the same change when modifying repo architecture, experiment schema, backend launcher contracts, build targets, or profiling command flow.
- Always update README/docs when changing build or profiling commands.
- Before adding a new helper, wrapper, or utility class/function, inspect `utils/` first and prefer existing shared wrappers such as CUDA memory/error helpers. If the helper can reasonably serve multiple ops, add or extend it in `utils/` instead of keeping it op-local.
- Public headers must live under a module-owned `include/kernel_lab/...` tree and be included with package paths such as `<kernel_lab/utils/cuda_utils.cuh>` or `<kernel_lab/ops/vector_add/config.hpp>`.
- Public C++/CUDA APIs must be namespaced under `kernel_lab`; operation APIs use `kernel_lab::ops::<op>`.
- Backend-local headers remain private quoted includes from the backend directory, for example `"kernels.cuh"`.
- Vendor headers remain private implementation details unless explicitly promoted to a public dependency.
- Do not delete or overwrite generated profiling data unless explicitly asked. Avoid unrelated refactors while tuning kernels; performance changes should be easy to isolate from build or CLI changes.
