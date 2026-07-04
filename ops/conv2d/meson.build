conv_arch = get_option('gpu_arch')
cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + conv_arch]
if get_option('enable_profiling_build')
    cuda_args += ['-O3']
endif
target_override = get_option('enable_profiling_build') ? ['buildtype=release'] : []
nvcc = find_program('nvcc', required: true)
cuobjdump = find_program('cuobjdump', required: false)

libops_conv2d_cuda = static_library(
    'ops_conv2d_cuda',
    'kernels/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    cuda_args: cuda_args,
    override_options: target_override)

libops_conv2d_cuda_dep = declare_dependency(
    link_with: libops_conv2d_cuda,
    include_directories: include_directories('kernels/cuda'))

executable('ops_conv2d_bench',
    'launcher/main.cu',
    dependencies: [cuda_utils_dep, libops_conv2d_cuda_dep],
    cuda_args: cuda_args,
    override_options: target_override)

# fastConv_1D / fastConv_2D have pre-existing issues (missing cuFFT linking,
# undefined variables). Wire them in here once fixed.
# executable('ops_conv2d_fft1d', 'kernels/cuda/fastConv_1D.cu', ...)
# executable('ops_conv2d_fft2d', 'kernels/cuda/fastConv_2D.cu', ...)

if get_option('enable_ptx_export')
    custom_target('ops_conv2d_ptx',
        input: 'kernels/cuda/kernels.cu',
        output: 'conv2d_kernels.ptx',
        command: [nvcc, '--ptx', '-arch=' + conv_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_conv2d_cubin',
        input: 'kernels/cuda/kernels.cu',
        output: 'conv2d_kernels.cubin',
        command: [nvcc, '--cubin', '-arch=' + conv_arch, '-lineinfo',
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
