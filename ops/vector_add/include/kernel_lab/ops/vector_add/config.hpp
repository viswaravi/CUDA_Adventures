#pragma once

#include <kernel_lab/utils/cli_args.hpp>
#include <kernel_lab/utils/experiment_types.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Operation-level config shared by vector_add backends. Backends validate the
// subset they implement, but they should accept the same normalized fields.
namespace kernel_lab::ops::vector_add {

struct VectorAddConfig {
  DType dtype = DType::Int32;
  std::string kernel = "scalar";
  std::string memory_host = "pageable";
  std::string execution_mode = "single_stream";
  std::uint64_t n = 1024ULL * 1024 * 256;
  float free_mem_threshold = 0.2f;
  float alpha = 1.0f;
  int streams = 1;
  int vector_width = 0;
  bool validate = true;
  int warmup = 5;
  int repeats = 20;
};

inline std::vector<ArgSpec> vectorAddArgs(
    const std::string &default_kernel = "scalar",
    const std::string &default_dtype = "int32") {
  return {
      {"--op", "string", "vector_add", "Operation name"},
      {"--n", "uint64", "536870912", "Array length (elements)"},
      {"--kernel", "string", default_kernel, "Vector-add kernel name"},
      {"--dtype", "string", default_dtype,
       "Data type, validated by the selected backend"},
      {"--memory-host", "string", "pageable", "Host memory mode"},
      {"--execution-mode", "string", "single_stream", "Execution mode"},
      {"--free-mem-threshold", "float", "0.2",
       "Fraction of free GPU memory to use per chunk (0, 1]"},
      {"--streams", "int", "1", "Number of concurrent CUDA streams"},
      {"--vector-width", "int", "0",
       "Vector width for vectorized CUDA kernel; 0 selects dtype default"},
      {"--alpha", "float", "1.0", "Scale factor for library reference paths"},
      {"--validate", "bool", "true", "Validate output"},
      {"--warmup", "int", "5", "Warm-up launches before measured repeats"},
      {"--repeats", "int", "20", "Measured launches or library calls"},
  };
}

inline VectorAddConfig vectorAddConfigFromRunConfig(
    const RunConfig &run_config,
    const std::string &default_dtype = "int32") {
  std::string op = get_string(run_config, "--op", "vector_add");
  if (op != "vector_add") {
    throw std::runtime_error("Unsupported --op for vector_add backend: " + op);
  }

  VectorAddConfig cfg;
  cfg.n = get_ull(run_config, "--n", 1024ULL * 1024 * 256);
  cfg.kernel = get_string(run_config, "--kernel", "scalar");
  cfg.dtype = parseDType(get_string(run_config, "--dtype", default_dtype));
  cfg.memory_host = get_string(run_config, "--memory-host", "pageable");
  cfg.execution_mode = get_string(run_config, "--execution-mode",
                                  "single_stream");
  cfg.free_mem_threshold = get_float(run_config, "--free-mem-threshold", 0.2f);
  cfg.alpha = get_float(run_config, "--alpha", 1.0f);
  cfg.streams = get_int(run_config, "--streams", 1);
  cfg.vector_width = get_int(run_config, "--vector-width", 0);
  cfg.validate = get_bool(run_config, "--validate", true);
  cfg.warmup = get_int(run_config, "--warmup", 5);
  cfg.repeats = get_int(run_config, "--repeats", 20);

  if (cfg.n == 0) {
    throw std::runtime_error("--n must be > 0");
  }
  if (cfg.streams < 1) {
    throw std::runtime_error("--streams must be >= 1");
  }
  if (cfg.free_mem_threshold <= 0.0f || cfg.free_mem_threshold > 1.0f) {
    throw std::runtime_error("--free-mem-threshold must be in (0, 1]");
  }
  
  return cfg;
}

} // namespace kernel_lab::ops::vector_add
