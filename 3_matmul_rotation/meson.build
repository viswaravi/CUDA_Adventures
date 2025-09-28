matmul_sources = ['matmul.cu']
rot_sources = ['rotation.cu']


executable(
    'matmul',
    matmul_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)

executable(
    'rotation',
    rot_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)

