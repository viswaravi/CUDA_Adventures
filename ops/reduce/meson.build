libops_reduce_cuda = static_library(
    'ops_reduce_cuda',
    'backends/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    override_options: cuda_target_options)

libops_reduce_cuda_dep = declare_dependency(
    link_with: libops_reduce_cuda,
    include_directories: include_directories('backends/cuda'))

executable('ops_reduce_cuda_runner',
    'backends/cuda/reduce_cuda_main.cu',
    dependencies: [cuda_utils_dep, libops_reduce_cuda_dep],
    override_options: cuda_target_options)

if get_option('enable_ptx_export')
    custom_target('ops_reduce_ptx',
        input: 'backends/cuda/kernels.cu',
        output: 'reduce_kernels.ptx',
        command: [nvcc, '--ptx'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_reduce_cubin',
        input: 'backends/cuda/kernels.cu',
        output: 'reduce_kernels.cubin',
        command: [nvcc, '--cubin'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)

    if get_option('enable_sass_export') and cuobjdump.found()
        custom_target('ops_reduce_sass',
            input: cubin_tgt,
            output: 'reduce_kernels.sass',
            command: [cuobjdump, '--dump-sass', '@INPUT@'],
            capture: true,
            build_by_default: true)
    endif
endif
