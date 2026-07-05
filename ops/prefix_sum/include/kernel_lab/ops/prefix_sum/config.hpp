#pragma once

#include <kernel_lab/utils/cli_args.hpp>
#include <kernel_lab/utils/experiment_types.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernel_lab::ops::prefix_sum {

struct PrefixSumConfig {
    DType dtype = DType::Float32;
    std::string kernel = "naive";
    std::uint64_t n = 1024;
    bool validate = true;
    int warmup = 0;
    int repeats = 1;
};

inline std::vector<ArgSpec> prefixSumArgs(const std::string &default_kernel = "naive",
                                          const std::string &default_dtype = "float32") {
    return {
        {"--op", "string", "prefix_sum", "Operation name"},
        {"--kernel", "string", default_kernel, "Prefix-sum kernel name"},
        {"--dtype", "string", default_dtype, "Data type, validated by the selected backend"},
        {"--n", "uint64", "1024", "Array length (elements)"},
        {"--validate", "bool", "true", "Validate output"},
        {"--warmup", "int", "0", "Warm-up scans before measured repeats"},
        {"--repeats", "int", "1", "Measured scan repetitions"},
    };
}

inline PrefixSumConfig prefixSumConfigFromRunConfig(const RunConfig &run_config,
                                                    const std::string &default_dtype = "float32") {
    std::string op = get_string(run_config, "--op", "prefix_sum");
    if (op != "prefix_sum") {
        throw std::runtime_error("Unsupported --op for prefix_sum backend: " + op);
    }

    PrefixSumConfig cfg;
    cfg.kernel = get_string(run_config, "--kernel", "naive");
    cfg.dtype = parseDType(get_string(run_config, "--dtype", default_dtype));
    cfg.n = get_ull(run_config, "--n", 1024ULL);
    cfg.validate = get_bool(run_config, "--validate", true);
    cfg.warmup = get_int(run_config, "--warmup", 0);
    cfg.repeats = get_int(run_config, "--repeats", 1);

    if (cfg.dtype != DType::Float32) {
        throw std::runtime_error("prefix_sum cuda supports only dtype float32");
    }
    if (cfg.n == 0) {
        throw std::runtime_error("--n must be > 0");
    }
    if (cfg.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    if (cfg.repeats < 1) {
        throw std::runtime_error("--repeats must be >= 1");
    }

    return cfg;
}

} // namespace kernel_lab::ops::prefix_sum
