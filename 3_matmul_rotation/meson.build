matmul_sources = ['matmul.cu']
rot_sources = ['rotation.cu']
matmul_arch = get_option('gpu_arch')
matmul_cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + matmul_arch]

matmul_target_override = []
if get_option('enable_profiling_build')
    matmul_target_override += ['buildtype=release']
endif


executable(
    'matmul',
    matmul_sources,
    dependencies: cuda_utils_dep,
    cuda_args: matmul_cuda_args,
    override_options: matmul_target_override,
)

executable(
    'rotation',
    rot_sources,
    dependencies: cuda_utils_dep,
    cuda_args: matmul_cuda_args,
    override_options: matmul_target_override,
)

