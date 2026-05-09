vec_add_sources = ['vec_add.cu']
vec_add_arch = 'sm_75'
vec_add_cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + vec_add_arch]
vec_add_cubin_args = ['--cubin'] + vec_add_cuda_args

executable(
    'vec_add',
    vec_add_sources,
    dependencies: cuda_utils_dep,
    cuda_args: vec_add_cuda_args,
)

nvcc = find_program('nvcc', required: true)

custom_target(
    'vec_add_cubin',
    input: vec_add_sources,
    output: 'vec_add.cubin',
    command: [nvcc] + vec_add_cubin_args + [
        '-I' + join_paths(meson.project_source_root(), 'utils'),
        '-o', '@OUTPUT@',
        '@INPUT@',
    ],
    build_by_default: true,
)