cuda_utils_sources = ['src/utils.cu']

cuda_utils_lib = static_library(
    'cuda_utils',
    cuda_utils_sources,
    include_directories: kernel_lab_utils_public_inc,
    override_options: cuda_target_options,
)

cuda_utils_dep = declare_dependency(
    link_with: cuda_utils_lib,
    include_directories: kernel_lab_utils_public_inc)
