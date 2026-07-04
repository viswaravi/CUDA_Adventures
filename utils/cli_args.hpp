#pragma once

#include <cctype>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// RunConfig – framework-level fields only, shared across ALL modules.
// Module-specific parameters (e.g. free_mem_threshold, num_heads) live in
// `extra` and are read back via the typed get_*() helpers below.
// ---------------------------------------------------------------------------
struct RunConfig {
    std::string variant; // selected variant name
    int device = 0;      // CUDA device id

    bool reset_device = true; // call cudaDeviceReset after run
    bool print_help = false;
    bool list_variants = false;

    // Module-specific key/value pairs populated from unknown --flags
    std::unordered_map<std::string, std::string> extra;
};

// ---------------------------------------------------------------------------
// ArgSpec – declaration of one module/variant-specific CLI flag.
// Modules declare these inside their VariantEntry so the parser can
// validate, populate defaults, and auto-generate usage text.
// ---------------------------------------------------------------------------
struct ArgSpec {
    std::string flag;        // e.g. "--free-mem-threshold"
    std::string type_hint;   // e.g. "float", "int", shown in usage
    std::string default_val; // e.g. "0.2"
    std::string description;
};

// ---------------------------------------------------------------------------
// VariantEntry – one experiment variant registered by each module
// ---------------------------------------------------------------------------
struct VariantEntry {
    std::string name;
    std::string description;
    std::vector<ArgSpec> args; // module-specific CLI flags for this variant
    std::function<void(const RunConfig &)> run;
};

using VariantRegistry = std::vector<VariantEntry>;

// ---------------------------------------------------------------------------
// Typed accessors for extra params – call from inside variant run lambdas
// ---------------------------------------------------------------------------
inline float get_float(const RunConfig &cfg, const std::string &flag, float default_val) {
    auto it = cfg.extra.find(flag);
    if (it == cfg.extra.end()) {
        return default_val;
    }
    try {
        return std::stof(it->second);
    } catch (...) {
        throw std::invalid_argument("Invalid float value for " + flag + ": " + it->second);
    }
}

inline int get_int(const RunConfig &cfg, const std::string &flag, int default_val) {
    auto it = cfg.extra.find(flag);
    if (it == cfg.extra.end()) {
        return default_val;
    }
    try {
        return std::stoi(it->second);
    } catch (...) {
        throw std::invalid_argument("Invalid int value for " + flag + ": " + it->second);
    }
}

inline std::string get_string(const RunConfig &cfg, const std::string &flag, const std::string &default_val) {
    auto it = cfg.extra.find(flag);
    if (it == cfg.extra.end()) {
        return default_val;
    }
    return it->second;
}

inline unsigned long long get_ull(const RunConfig &cfg, const std::string &flag, unsigned long long default_val) {
    auto it = cfg.extra.find(flag);
    if (it == cfg.extra.end()) {
        return default_val;
    }
    try {
        return std::stoull(it->second);
    } catch (...) {
        throw std::invalid_argument("Invalid value for " + flag + ": " + it->second);
    }
}

inline bool get_bool(const RunConfig &cfg, const std::string &flag, bool default_val) {
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

    throw std::invalid_argument("Invalid bool value for " + flag + ": " + it->second);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
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

inline const VariantEntry *find_variant(const VariantRegistry &variants, const std::string &name) {
    for (const auto &v : variants) {
        if (v.name == name) {
            return &v;
        }
    }
    return nullptr;
}

} // namespace cli_detail

// ---------------------------------------------------------------------------
// print_usage
// ---------------------------------------------------------------------------
inline void print_usage(const char *program, const VariantRegistry &variants) {
    std::cout << "Usage: " << program << " [options]\n\n"
              << "Global options:\n"
              << "  --variant <name>    Variant to run (see --list-variants)\n"
              << "  --device <id>       CUDA device id (default: 0)\n"
              << "  --no-reset          Skip cudaDeviceReset (useful inside profiler "
                 "loops)\n"
              << "  --list-variants     List supported variants and their defaults\n"
              << "  --help              Show this message\n\n"
              << "Variants and their specific options:\n";

    for (const auto &v : variants) {
        std::cout << "  " << v.name << "\n"
                  << "      " << v.description << "\n";
        for (const auto &a : v.args) {
            std::ostringstream line;
            line << "      " << a.flag << " <" << a.type_hint << ">";
            std::string prefix = line.str();
            // Pad to column 40
            for (int i = static_cast<int>(prefix.size()); i < 40; ++i) {
                std::cout << ' ';
            }
            std::cout << prefix << "  " << a.description << "  (default: " << a.default_val << ")\n";
        }
        std::cout << "\n";
    }
}

inline void print_variants(const VariantRegistry &variants) {
    std::cout << "Supported variants:\n";
    for (const auto &v : variants) {
        std::cout << "  " << v.name << "\n";
    }
}

// ---------------------------------------------------------------------------
// parse_args
// Two-phase:
//   1. Collect all flags; known globals go into RunConfig fields,
//      unknown --key value pairs go into a raw extras map.
//   2. Look up the selected variant and validate that every key in the
//      raw extras map is declared in that variant's ArgSpec list.
//      Fill missing ArgSpecs with their defaults into cfg.extra.
// ---------------------------------------------------------------------------
inline RunConfig parse_args(int argc, char **argv, const VariantRegistry &variants) {
    RunConfig cfg;

    if (!variants.empty()) {
        cfg.variant = variants.front().name;
    }

    // Phase 1: consume flags
    std::unordered_map<std::string, std::string> raw_extra;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help") {
            cfg.print_help = true;
            continue;
        }
        if (arg == "--list-variants") {
            cfg.list_variants = true;
            continue;
        }
        if (arg == "--no-reset") {
            cfg.reset_device = false;
            continue;
        }

        if (arg == "--variant") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for argument: " + arg);
            }
            const std::string value = argv[++i];
            cfg.variant = value;
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for argument: " + arg);
            }
            const std::string value = argv[++i];
            if (!cli_detail::parseInt(value, cfg.device)) {
                throw std::invalid_argument("Invalid --device value: " + value);
            }
        } else if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
            // Unknown flag → stash for variant-specific validation.
            // If no value follows (or the next token is another flag), treat as
            // implicit boolean true and validate in phase 2.
            std::string value = cli_detail::kImplicitBoolTrueSentinel;
            if (i + 1 < argc) {
                const std::string next = argv[i + 1];
                if (!(next.size() > 1 && next[0] == '-' && next[1] == '-')) {
                    value = next;
                    ++i;
                }
            }
            raw_extra[arg] = value;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    // Phase 2: resolve variant, validate/fill extra args
    if (!cfg.print_help && !cfg.list_variants) {
        const VariantEntry *entry = cli_detail::find_variant(variants, cfg.variant);
        if (!entry) {
            std::string known;
            for (const auto &v : variants) {
                known += "\n  " + v.name;
            }
            throw std::invalid_argument("Unknown --variant: '" + cfg.variant + "'\nKnown variants:" + known);
        }

        // Validate extra flags against this variant's ArgSpec
        for (const auto &kv : raw_extra) {
            bool declared = false;
            for (const auto &spec : entry->args) {
                if (spec.flag == kv.first) {
                    declared = true;
                    break;
                }
            }
            if (!declared) {
                throw std::invalid_argument("Flag " + kv.first + " is not valid for variant '" + cfg.variant + "'");
            }
        }

        // Populate cfg.extra: user value if provided, else variant default
        for (const auto &spec : entry->args) {
            auto it = raw_extra.find(spec.flag);
            const std::string value = (it != raw_extra.end()) ? it->second : spec.default_val;

            if (cli_detail::is_bool_type_hint(spec.type_hint)) {
                const std::string bool_value = (value == cli_detail::kImplicitBoolTrueSentinel) ? "true" : value;
                bool parsed = false;
                if (!cli_detail::parseBool(bool_value, parsed)) {
                    throw std::invalid_argument("Invalid boolean value for " + spec.flag + ": " + bool_value +
                                                " (expected true/false, 1/0, yes/no, on/off)");
                }
                cfg.extra[spec.flag] = parsed ? "true" : "false";
            } else {
                if (it != raw_extra.end() && it->second == cli_detail::kImplicitBoolTrueSentinel) {
                    throw std::invalid_argument("Missing value for argument: " + spec.flag);
                }
                cfg.extra[spec.flag] = value;
            }
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// run_variant
// ---------------------------------------------------------------------------
inline void run_variant(const RunConfig &cfg, const VariantRegistry &variants) {
    const VariantEntry *entry = cli_detail::find_variant(variants, cfg.variant);
    if (!entry) {
        std::string known;
        for (const auto &v : variants) {
            known += "\n  " + v.name;
        }
        throw std::invalid_argument("Unknown --variant: '" + cfg.variant + "'\nKnown variants:" + known);
    }
    entry->run(cfg);
}
