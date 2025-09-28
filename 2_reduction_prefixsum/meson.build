reduction_sources = ['reduction.cu']
prefixsum_sources = ['prefixSum.cu']


executable(
    'reduction',
    reduction_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)

executable(
    'prefixSum',
    prefixsum_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)

