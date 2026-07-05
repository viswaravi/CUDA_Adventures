ops_rotation_private_vendor_inc = include_directories('../../utils')

libops_rotation_cuda = static_library(
    'ops_rotation_cuda',
    'backends/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    override_options: cuda_target_options)

libops_rotation_cuda_dep = declare_dependency(
    link_with: libops_rotation_cuda,
    include_directories: include_directories('backends/cuda'))

executable('ops_rotation_cuda_runner',
    'backends/cuda/rotation_cuda_main.cu',
    dependencies: [cuda_utils_dep, libops_rotation_cuda_dep],
    include_directories: ops_rotation_private_vendor_inc,
    override_options: cuda_target_options)

if get_option('enable_ptx_export')
    custom_target('ops_rotation_ptx',
        input: 'backends/cuda/kernels.cu',
        output: 'rotation_kernels.ptx',
        command: [nvcc, '--ptx'] + cuda_export_args + [
                  kernel_lab_utils_public_include_arg,
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_rotation_cubin',
        input: 'backends/cuda/kernels.cu',
        output: 'rotation_kernels.cubin',
        command: [nvcc, '--cubin'] + cuda_export_args + [
                  kernel_lab_utils_public_include_arg,
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)

    if get_option('enable_sass_export') and cuobjdump.found()
        custom_target('ops_rotation_sass',
            input: cubin_tgt,
            output: 'rotation_kernels.sass',
            command: [cuobjdump, '--dump-sass', '@INPUT@'],
            capture: true,
            build_by_default: true)
    endif
endif
