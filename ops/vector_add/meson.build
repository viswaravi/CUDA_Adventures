cublas_dep = meson.get_compiler('cpp').find_library('cublas', required: false)
ops_vector_add_public_inc = include_directories('include')
ops_vector_add_public_include_arg = '-I' + join_paths(meson.current_source_dir(), 'include')
ops_vector_add_cuda_private_inc = include_directories('backends/cuda')

# Kernel static library — contains __global__ kernels only, no main()
libops_vector_add_cuda = static_library(
    'ops_vector_add_cuda',
    'backends/cuda/kernels.cu',
    dependencies: cuda_utils_dep,
    override_options: cuda_target_options)

libops_vector_add_cuda_dep = declare_dependency(
    link_with: libops_vector_add_cuda,
    include_directories: [ops_vector_add_public_inc, ops_vector_add_cuda_private_inc])

# CUDA runner binary - to test the kernels in the static library
# Backend - CUDA
executable('ops_vector_add_cuda_runner',
    'backends/cuda/vector_add_cuda_runner.cu',
    dependencies: [cuda_utils_dep, libops_vector_add_cuda_dep],
    override_options: cuda_target_options)

# PTX export (opt-in: -Denable_ptx_export=true)
if get_option('enable_ptx_export')
    custom_target('ops_vector_add_ptx',
        input: 'backends/cuda/kernels.cu',
        output: 'vector_add_kernels.ptx',
        command: [nvcc, '--ptx'] + cuda_export_args + [
                  kernel_lab_utils_public_include_arg,
                  ops_vector_add_public_include_arg,
                  '-o', '@OUTPUT@', '@INPUT@'],
        build_by_default: true)
endif

# Cubin export (opt-in: -Denable_cubin_export=true)
if get_option('enable_cubin_export')
    cubin_tgt = custom_target('ops_vector_add_cubin',
        input: 'backends/cuda/kernels.cu',
        output: 'vector_add_kernels.cubin',
        command: [nvcc, '--cubin'] + cuda_export_args + [
                  kernel_lab_utils_public_include_arg,
                  ops_vector_add_public_include_arg,
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

# Backend - CUBLAS
executable('ops_vector_add_cublas_runner',
    'backends/cublas/vector_add_cublas_main.cpp',
    dependencies: [cuda_utils_dep, cublas_dep],
    include_directories: ops_vector_add_public_inc)
