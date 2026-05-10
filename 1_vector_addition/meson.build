# Vector Addition sweep configuration (local to this module)
vec_add_n_small = 67108864        # 64 MiB
vec_add_n_medium = 268435456      # 256 MiB
vec_add_n_large = 536870912       # 512 MiB

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

# ============ Run targets (quick manual testing) ============
n_small_str = '@0@'.format(vec_add_n_small)
n_medium_str = '@0@'.format(vec_add_n_medium)
n_large_str = '@0@'.format(vec_add_n_large)

run_target(
  'run_vec_add_pageable',
  command: [vec_add_exe, '--variant', 'pageable', '--n', n_medium_str],
)

run_target(
  'run_vec_add_pinned',
  command: [vec_add_exe, '--variant', 'pinned', '--n', n_medium_str],
)

run_target(
  'run_vec_add_pageable_large',
  command: [
    vec_add_exe,
    '--variant', 'pageable-large',
    '--n', n_large_str,
    '--free-mem-threshold', '0.2',
  ],
)

run_target(
  'run_vec_add_streamed_large',
  command: [
    vec_add_exe,
    '--variant', 'streamed-large',
    '--n', n_large_str,
    '--free-mem-threshold', '0.2',
    '--streams', '4',
  ],
)

# ============ Profiling targets (NCU, NSYS) ============
python3 = find_program('python3', required: true)
profile_runner = files('../tools/profile_runner.py')
vec_add_bin_path = join_paths(meson.current_build_dir(), 'vec_add')

profile_mode = 'dev'
if get_option('enable_profiling_build')
  profile_mode = 'profile'
endif

# NCU targets
ncu = find_program('ncu', required: false)
if ncu.found()
  run_target(
    'profile_vec_add_ncu_pageable',
    command: [
      python3,
      profile_runner,
      '--tool', 'ncu',
      '--profiler-bin', ncu,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling', '1', 'ncu', 'pageable'),
      '--report-stem', 'pageable__n' + n_small_str + '__arch' + vec_add_arch + '__build' + profile_mode,
      '--',
      '--variant', 'pageable',
      '--n', n_small_str,
    ],
  )

  run_target(
    'profile_vec_add_ncu_pinned',
    command: [
      python3,
      profile_runner,
      '--tool', 'ncu',
      '--profiler-bin', ncu,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling', '1', 'ncu', 'pinned'),
      '--report-stem', 'pinned__n' + n_medium_str + '__arch' + vec_add_arch + '__build' + profile_mode,
      '--',
      '--variant', 'pinned',
      '--n', n_medium_str,
    ],
  )

  run_target(
    'profile_vec_add_ncu_streamed_large',
    command: [
      python3,
      profile_runner,
      '--tool', 'ncu',
      '--profiler-bin', ncu,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling', '1', 'ncu', 'streamed-large'),
      '--report-stem', 'streamed-large__n' + n_large_str + '__arch' + vec_add_arch + '__build' + profile_mode,
      '--',
      '--variant', 'streamed-large',
      '--n', n_large_str,
      '--streams', '4',
    ],
  )
endif

# NSYS targets
nsys = find_program('nsys', required: false)
if nsys.found()
  run_target(
    'profile_vec_add_nsys_pageable',
    command: [
      python3,
      profile_runner,
      '--tool', 'nsys',
      '--profiler-bin', nsys,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling', '1', 'nsys', 'pageable'),
      '--report-stem', 'pageable__n' + n_small_str + '__arch' + vec_add_arch + '__build' + profile_mode,
      '--',
      '--variant', 'pageable',
      '--n', n_small_str,
    ],
  )

  run_target(
    'profile_vec_add_nsys_streamed_large',
    command: [
      python3,
      profile_runner,
      '--tool', 'nsys',
      '--profiler-bin', nsys,
      '--binary', vec_add_bin_path,
      '--output-dir', join_paths(meson.project_source_root(), 'profiling', '1', 'nsys', 'streamed-large'),
      '--report-stem', 'streamed-large__n' + n_large_str + '__arch' + vec_add_arch + '__build' + profile_mode,
      '--',
      '--variant', 'streamed-large',
      '--n', n_large_str,
      '--streams', '4',
    ],
  )
endif
