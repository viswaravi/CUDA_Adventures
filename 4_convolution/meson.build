conv_sources = ['convolution.cu']
fconv1D_sources = ['fastConv_1D.cu']
fconv2D_sources = ['fastConv_2D.cu']

cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + get_option('gpu_arch')]
if get_option('enable_profiling_build')
    cuda_args += ['-O3']
endif

executable(
    'convolution',
    conv_sources,
    dependencies: cuda_utils_dep,
    cuda_args: cuda_args,
)

# Note: fconv1D and fconv2D have pre-existing build issues (missing cuFFT linking, undefined variables)
# Commenting them out to allow the convolution target to build successfully
# TODO: Fix cuFFT linking and variable scoping in fastConv_1D/2D sources
#
# executable(
#     'fconv1D',
#     fconv1D_sources,
#     dependencies: cuda_utils_dep,
#     cuda_args: cuda_args,
# )
#
# executable(
#     'fconv2D',
#     fconv2D_sources,
#     dependencies: cuda_utils_dep,
#     cuda_args: cuda_args,
# )