conv_sources = ['convolution.cu']
fconv1D_sources = ['fastconv_1D.cu']
fconv2D_sources = ['fastConv_2D.cu']

executable(
    'convolution',
    conv_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)

executable(
    'fconv1D',
    fconv1D_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)

executable(
    'fconv2D',
    fconv2D_sources,
    dependencies: cuda_utils_dep,
    cuda_args: ['--use_fast_math'],
)