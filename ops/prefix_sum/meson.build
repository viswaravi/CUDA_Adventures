ps_arch = get_option('gpu_arch')
cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + ps_arch]
target_override = get_option('enable_profiling_build') ? ['buildtype=release'] : []
nvcc = find_program('nvcc', required: true)
cuobjdump = find_program('cuobjdump', required: false)

libops_prefix_sum_cuda = static_library(
    'ops_prefix_sum_cuda',
    'kernels/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    cuda_args: cuda_args,
    override_options: target_override)

libops_prefix_sum_cuda_dep = declare_dependency(
    link_with: libops_prefix_sum_cuda,
    include_directories: include_directories('kernels/cuda'))

executable('ops_prefix_sum_bench',
    'launcher/main.cu',
    dependencies: [cuda_utils_dep, libops_prefix_sum_cuda_dep],
    cuda_args: cuda_args,
    override_options: target_override)

if get_option('enable_ptx_export')
    custom_target('ops_prefix_sum_ptx',
        input: 'kernels/cuda/kernels.cu',
        output: 'prefix_sum_kernels.ptx',
        command: [nvcc, '--ptx', '-arch=' + ps_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_prefix_sum_cubin',
        input: 'kernels/cuda/kernels.cu',
        output: 'prefix_sum_kernels.cubin',
        command: [nvcc, '--cubin', '-arch=' + ps_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)

    if get_option('enable_sass_export') and cuobjdump.found()
        custom_target('ops_prefix_sum_sass',
            input: cubin_tgt,
            output: 'prefix_sum_kernels.sass',
            command: [cuobjdump, '--dump-sass', '@INPUT@'],
            capture: true,
            build_by_default: true)
    endif
endif
