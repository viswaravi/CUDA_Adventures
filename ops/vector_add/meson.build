va_arch = get_option('gpu_arch')
cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + va_arch]
target_override = get_option('enable_profiling_build') ? ['buildtype=release'] : []
nvcc = find_program('nvcc', required: true)
cuobjdump = find_program('cuobjdump', required: false)

# 1. Kernel static library — contains __global__ kernels only, no main()
libops_vector_add_cuda = static_library(
    'ops_vector_add_cuda',
    'kernels/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    cuda_args: cuda_args,
    override_options: target_override)

libops_vector_add_cuda_dep = declare_dependency(
    link_with: libops_vector_add_cuda,
    include_directories: include_directories('kernels/cuda'))

# 2. Manual launcher binary — driven by profile_matrix_runner.py
executable('ops_vector_add_bench',
    'launcher/main.cu',
    dependencies: [cuda_utils_dep, libops_vector_add_cuda_dep],
    cuda_args: cuda_args,
    override_options: target_override)

# 3. PTX export (opt-in: -Denable_ptx_export=true)
if get_option('enable_ptx_export')
    custom_target('ops_vector_add_ptx',
        input: 'kernels/cuda/kernels.cu',
        output: 'vector_add_kernels.ptx',
        command: [nvcc, '--ptx', '-arch=' + va_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

# 4. Cubin export (opt-in: -Denable_cubin_export=true)
if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_vector_add_cubin',
        input: 'kernels/cuda/kernels.cu',
        output: 'vector_add_kernels.cubin',
        command: [nvcc, '--cubin', '-arch=' + va_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)

    if get_option('enable_sass_export') and cuobjdump.found()
        custom_target('ops_vector_add_sass',
            input: cubin_tgt,
            output: 'vector_add_kernels.sass',
            command: [cuobjdump, '--dump-sass', '@INPUT@'],
            capture: true,
            build_by_default: true)
    endif
endif
