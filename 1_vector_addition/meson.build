vec_add_sources = ['vec_add.cu']
vec_add_arch = get_option('gpu_arch')
vec_add_cuda_args = ['--use_fast_math', '-lineinfo', '-arch=' + vec_add_arch]
vec_add_cubin_args = ['--cubin'] + vec_add_cuda_args

vec_add_target_override = []
if get_option('enable_profiling_build')
  vec_add_target_override += ['buildtype=release']
endif

vec_add_exe = executable(
  'vec_add',
  vec_add_sources,
  dependencies: cuda_utils_dep,
  cuda_args: vec_add_cuda_args,
  override_options: vec_add_target_override,
)

# Export cubin for inspection
nvcc = find_program('nvcc', required: true)
custom_target(
  'vec_add_cubin',
  input: vec_add_sources,
  output: 'vec_add.cubin',
  command: [nvcc] + vec_add_cubin_args + [
    '-I' + join_paths(meson.project_source_root(), 'utils'),
    '-o', '@OUTPUT@',
    '@INPUT@',
  ],
  build_by_default: true,
)

python3 = find_program(join_paths(meson.project_source_root(), '.venv', 'bin', 'python'), required: true)
vec_add_bin_path = join_paths(meson.current_build_dir(), 'vec_add')
profile_matrix_runner = files('../tools/profile_matrix_runner.py')
vec_add_matrix_file = join_paths(meson.project_source_root(), '1_vector_addition', 'vec_add_experiments.yaml')

profile_mode = 'dev'
if get_option('enable_profiling_build')
  profile_mode = 'profile'
endif

ncu = find_program('ncu', required: false)
if ncu.found()
  run_target(
    'profile_vec_add_ncu_sweep',
    command: [
      python3,
      profile_matrix_runner,
      '--matrix', vec_add_matrix_file,
      '--tool', 'ncu',
      '--profiler-bin', ncu,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling'),
      '--build-tag', profile_mode,
    ],
  )
endif

nsys = find_program('nsys', required: false)
if nsys.found()
  run_target(
    'profile_vec_add_nsys_sweep',
    command: [
      python3,
      profile_matrix_runner,
      '--matrix', vec_add_matrix_file,
      '--tool', 'nsys',
      '--profiler-bin', nsys,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling'),
      '--build-tag', profile_mode,
    ],
  )
endif
