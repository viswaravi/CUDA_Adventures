reduction_sources = ['reduction.cu']
prefixsum_sources = ['prefixSum.cu']
reduction_arch = get_option('gpu_arch')
reduction_cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + reduction_arch]

reduction_target_override = []
if get_option('enable_profiling_build')
    reduction_target_override += ['buildtype=release']
endif


executable(
    'reduction',
    reduction_sources,
    dependencies: cuda_utils_dep,
        cuda_args: reduction_cuda_args,
        override_options: reduction_target_override,
)

executable(
    'prefixSum',
    prefixsum_sources,
    dependencies: cuda_utils_dep,
        cuda_args: reduction_cuda_args,
        override_options: reduction_target_override,
)




# Export cubin for inspection
nvcc = find_program('nvcc', required: true)
reduction_cubin_args = ['--cubin'] + reduction_cuda_args

custom_target(
    'reduction_cubin',
    input: reduction_sources,
    output: 'reduction.cubin',
    command: [nvcc] + reduction_cubin_args + [
        '-I' + join_paths(meson.project_source_root(), 'utils'),
        '-o', '@OUTPUT@',
        '@INPUT@',
    ],
    build_by_default: true,
)

custom_target(
    'prefixSum_cubin',
    input: prefixsum_sources,
    output: 'prefixSum.cubin',
    command: [nvcc] + reduction_cubin_args + [
        '-I' + join_paths(meson.project_source_root(), 'utils'),
        '-o', '@OUTPUT@',
        '@INPUT@',
    ],
    build_by_default: true,
)