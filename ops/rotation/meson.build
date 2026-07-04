rot_arch = get_option('gpu_arch')
cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + rot_arch]
target_override = get_option('enable_profiling_build') ? ['buildtype=release'] : []
nvcc = find_program('nvcc', required: true)
cuobjdump = find_program('cuobjdump', required: false)

libops_rotation_cuda = static_library(
    'ops_rotation_cuda',
    'kernels/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    cuda_args: cuda_args,
    override_options: target_override)

libops_rotation_cuda_dep = declare_dependency(
    link_with: libops_rotation_cuda,
    include_directories: include_directories('kernels/cuda'))

executable('ops_rotation_bench',
    'launcher/main.cu',
    dependencies: [cuda_utils_dep, libops_rotation_cuda_dep],
    cuda_args: cuda_args,
    override_options: target_override)

if get_option('enable_ptx_export')
    custom_target('ops_rotation_ptx',
        input: 'kernels/cuda/kernels.cu',
        output: 'rotation_kernels.ptx',
        command: [nvcc, '--ptx', '-arch=' + rot_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_rotation_cubin',
        input: 'kernels/cuda/kernels.cu',
        output: 'rotation_kernels.cubin',
        command: [nvcc, '--cubin', '-arch=' + rot_arch, '-lineinfo',
                  '-I' + join_paths(meson.project_source_root(), 'utils'),
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
