#pragma once

#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Shared normalized CLI parser for backend-owned experiment launchers.
//
// Launchers declare their accepted flags with ArgSpec, parse argv into
// RunConfig, then convert the string values into their operation-specific
// config structs using the typed get_* helpers below.

struct RunConfig {
  int device = 0;
  bool reset_device = true;
  bool print_help = false;
  std::unordered_map<std::string, std::string> extra;
};

struct ArgSpec {
  std::string flag;
  std::string type_hint;
  std::string default_val;
  std::string description;
};

inline float get_float(const RunConfig &cfg, const std::string &flag,
                       float default_val) {
  auto it = cfg.extra.find(flag);
  if (it == cfg.extra.end()) {
    return default_val;
  }
  try {
    return std::stof(it->second);
  } catch (...) {
    throw std::invalid_argument("Invalid float value for " + flag + ": " +
                                it->second);
  }
}

inline int get_int(const RunConfig &cfg, const std::string &flag,
                   int default_val) {
  auto it = cfg.extra.find(flag);
  if (it == cfg.extra.end()) {
    return default_val;
  }
  try {
    return std::stoi(it->second);
  } catch (...) {
    throw std::invalid_argument("Invalid int value for " + flag + ": " +
                                it->second);
  }
}

inline std::string get_string(const RunConfig &cfg, const std::string &flag,
                              const std::string &default_val) {
  auto it = cfg.extra.find(flag);
  if (it == cfg.extra.end()) {
    return default_val;
  }
  return it->second;
}

inline unsigned long long get_ull(const RunConfig &cfg,
                                  const std::string &flag,
                                  unsigned long long default_val) {
  auto it = cfg.extra.find(flag);
  if (it == cfg.extra.end()) {
    return default_val;
  }
  try {
    return std::stoull(it->second);
  } catch (...) {
    throw std::invalid_argument("Invalid value for " + flag + ": " +
                                it->second);
  }
}

inline bool get_bool(const RunConfig &cfg, const std::string &flag,
                     bool default_val) {
  auto it = cfg.extra.find(flag);
  if (it == cfg.extra.end()) {
    return default_val;
  }

  std::string value = it->second;
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }

  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }

  throw std::invalid_argument("Invalid bool value for " + flag + ": " +
                              it->second);
}

namespace cli_detail {
static const char *kImplicitBoolTrueSentinel = "__cli_implicit_bool_true__";

inline bool parseInt(const std::string &s, int &out) {
  try {
    size_t p = 0;
    out = std::stoi(s, &p);
    return p == s.size();
  } catch (...) {
    return false;
  }
}

inline bool parseBool(const std::string &s, bool &out) {
  std::string value = s;
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }

  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    out = false;
    return true;
  }
  return false;
}

inline bool is_bool_type_hint(const std::string &type_hint) {
  std::string hint = type_hint;
  for (char &ch : hint) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return hint == "bool" || hint == "boolean";
}

inline const ArgSpec *find_arg(const std::vector<ArgSpec> &args,
                               const std::string &flag) {
  for (const auto &arg : args) {
    if (arg.flag == flag) {
      return &arg;
    }
  }
  return nullptr;
}

} // namespace cli_detail

inline void print_usage(const char *program, const std::vector<ArgSpec> &args,
                        const std::string &description = "") {
  std::cout << "Usage: " << program << " [options]\n\n";
  if (!description.empty()) {
    std::cout << description << "\n\n";
  }

  std::cout << "Global options:\n"
            << "  --device <id>       CUDA device id (default: 0)\n"
            << "  --no-reset          Skip cudaDeviceReset\n"
            << "  --help              Show this message\n\n"
            << "Backend options:\n";

  for (const auto &arg : args) {
    std::ostringstream line;
    line << "  " << arg.flag << " <" << arg.type_hint << ">";
    std::string prefix = line.str();
    std::cout << prefix;
    for (int i = static_cast<int>(prefix.size()); i < 34; ++i) {
      std::cout << ' ';
    }
    std::cout << "  " << arg.description << "  (default: " << arg.default_val
              << ")\n";
  }
}

inline RunConfig parse_args(int argc, char **argv,
                            const std::vector<ArgSpec> &args) {
  RunConfig cfg;
  std::unordered_map<std::string, std::string> raw_extra;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help") {
      cfg.print_help = true;
      continue;
    }
    if (arg == "--no-reset") {
      cfg.reset_device = false;
      continue;
    }
    if (arg == "--device") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for argument: " + arg);
      }
      const std::string value = argv[++i];
      if (!cli_detail::parseInt(value, cfg.device)) {
        throw std::invalid_argument("Invalid --device value: " + value);
      }
      continue;
    }
    if (arg == "--variant" || arg == "--list-variants") {
      throw std::invalid_argument(
          arg + " is no longer supported; use normalized backend flags");
    }
    if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
      std::string value = cli_detail::kImplicitBoolTrueSentinel;
      if (i + 1 < argc) {
        const std::string next = argv[i + 1];
        if (!(next.size() > 1 && next[0] == '-' && next[1] == '-')) {
          value = next;
          ++i;
        }
      }
      raw_extra[arg] = value;
      continue;
    }

    throw std::invalid_argument("Unknown argument: " + arg);
  }

  if (cfg.print_help) {
    return cfg;
  }

  for (const auto &kv : raw_extra) {
    if (cli_detail::find_arg(args, kv.first) == nullptr) {
      throw std::invalid_argument("Unknown backend flag: " + kv.first);
    }
  }

  for (const auto &spec : args) {
    auto it = raw_extra.find(spec.flag);
    const std::string value =
        (it != raw_extra.end()) ? it->second : spec.default_val;

    if (cli_detail::is_bool_type_hint(spec.type_hint)) {
      const std::string bool_value =
          (value == cli_detail::kImplicitBoolTrueSentinel) ? "true" : value;
      bool parsed = false;
      if (!cli_detail::parseBool(bool_value, parsed)) {
        throw std::invalid_argument("Invalid boolean value for " + spec.flag +
                                    ": " + bool_value);
      }
      cfg.extra[spec.flag] = parsed ? "true" : "false";
      continue;
    }

    if (it != raw_extra.end() &&
        it->second == cli_detail::kImplicitBoolTrueSentinel) {
      throw std::invalid_argument("Missing value for argument: " + spec.flag);
    }
    cfg.extra[spec.flag] = value;
  }

  return cfg;
}
