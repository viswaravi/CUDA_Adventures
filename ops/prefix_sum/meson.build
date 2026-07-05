libops_prefix_sum_cuda = static_library(
    'ops_prefix_sum_cuda',
    'backends/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    override_options: cuda_target_options)

libops_prefix_sum_cuda_dep = declare_dependency(
    link_with: libops_prefix_sum_cuda,
    include_directories: include_directories('backends/cuda'))

executable('ops_prefix_sum_cuda_runner',
    'backends/cuda/prefix_sum_cuda_main.cu',
    dependencies: [cuda_utils_dep, libops_prefix_sum_cuda_dep],
    override_options: cuda_target_options)

if get_option('enable_ptx_export')
    custom_target('ops_prefix_sum_ptx',
        input: 'backends/cuda/kernels.cu',
        output: 'prefix_sum_kernels.ptx',
        command: [nvcc, '--ptx'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_prefix_sum_cubin',
        input: 'backends/cuda/kernels.cu',
        output: 'prefix_sum_kernels.cubin',
        command: [nvcc, '--cubin'] + cuda_export_args + [
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
