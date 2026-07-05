cuda_utils_sources = ['utils.cu']
cuda_utils_inc = include_directories('.')

cuda_utils_lib = static_library(
    'cuda_utils',
    cuda_utils_sources,
    include_directories: cuda_utils_inc,
    override_options: cuda_target_options,
)

cuda_utils_dep = declare_dependency(link_with: cuda_utils_lib, include_directories: cuda_utils_inc)
