vec_add_sources = ['vec_add.cu']

executable(
    'vec_add',
    vec_add_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)