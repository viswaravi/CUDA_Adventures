project(
  'cuda-multi-targets',
  'cpp',
  'cuda',
  version: '1.0',
  default_options: ['cpp_std=c++17', 'cuda_std=c++17'],
)

cuda_project_args = ['-arch=' + get_option('gpu_arch')]
if get_option('enable_cuda_fast_math')
  cuda_project_args += ['--use_fast_math']
endif
if get_option('enable_cuda_lineinfo')
  cuda_project_args += ['-lineinfo']
endif

add_project_arguments(cuda_project_args, language: 'cuda')

cuda_export_args = ['-arch=' + get_option('gpu_arch')]
if get_option('enable_cuda_lineinfo')
  cuda_export_args += ['-lineinfo']
endif

cuda_target_options = ['warning_level=0']
nvcc = find_program('nvcc', required: true)
cuobjdump = find_program('cuobjdump', required: false)

subdir('utils')
subdir('ops')
