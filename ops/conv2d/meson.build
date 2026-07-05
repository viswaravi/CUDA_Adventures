libops_conv2d_cuda = static_library(
    'ops_conv2d_cuda',
    'backends/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    override_options: cuda_target_options)

libops_conv2d_cuda_dep = declare_dependency(
    link_with: libops_conv2d_cuda,
    include_directories: include_directories('backends/cuda'))

executable('ops_conv2d_cuda_runner',
    'backends/cuda/conv2d_cuda_main.cu',
    dependencies: [cuda_utils_dep, libops_conv2d_cuda_dep],
    override_options: cuda_target_options)

# fastConv_1D / fastConv_2D have pre-existing issues (missing cuFFT linking,
# undefined variables). Wire them in here once fixed.
# executable('ops_conv2d_fft1d', 'backends/cuda/fastConv_1D.cu', ...)
# executable('ops_conv2d_fft2d', 'backends/cuda/fastConv_2D.cu', ...)

if get_option('enable_ptx_export')
    custom_target('ops_conv2d_ptx',
        input: 'backends/cuda/kernels.cu',
        output: 'conv2d_kernels.ptx',
        command: [nvcc, '--ptx'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_conv2d_cubin',
        input: 'backends/cuda/kernels.cu',
        output: 'conv2d_kernels.cubin',
        command: [nvcc, '--cubin'] + cuda_export_args + [
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)

    if get_option('enable_sass_export') and cuobjdump.found()
        custom_target('ops_conv2d_sass',
            input: cubin_tgt,
            output: 'conv2d_kernels.sass',
            command: [cuobjdump, '--dump-sass', '@INPUT@'],
            capture: true,
            build_by_default: true)
    endif
endif
