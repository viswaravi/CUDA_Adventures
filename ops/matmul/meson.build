# TODO: backends/cuda/wmma.cu and backends/cuda/matvec.cu (moved from 5_tensors) duplicate
# CudaMemory<T> and CUDA_CALL internally instead of linking cuda_utils_dep.
# Fix the includes in those files, then add them to a separate static lib or
# extend this one with an explicit 'wmma' variant.

libops_matmul_cuda = static_library(
    'ops_matmul_cuda',
    'backends/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    override_options: cuda_target_options)

libops_matmul_cuda_dep = declare_dependency(
    link_with: libops_matmul_cuda,
    include_directories: include_directories('backends/cuda'))

executable('ops_matmul_cuda_runner',
    'backends/cuda/matmul_cuda_main.cu',
    dependencies: [cuda_utils_dep, libops_matmul_cuda_dep],
    override_options: cuda_target_options)

if get_option('enable_ptx_export')
    custom_target('ops_matmul_ptx',
        input: 'backends/cuda/kernels.cu',
        output: 'matmul_kernels.ptx',
        command: [nvcc, '--ptx'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_matmul_cubin',
        input: 'backends/cuda/kernels.cu',
        output: 'matmul_kernels.cubin',
        command: [nvcc, '--cubin'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)

    if get_option('enable_sass_export') and cuobjdump.found()
        custom_target('ops_matmul_sass',
            input: cubin_tgt,
            output: 'matmul_kernels.sass',
            command: [cuobjdump, '--dump-sass', '@INPUT@'],
            capture: true,
            build_by_default: true)
    endif
endif
