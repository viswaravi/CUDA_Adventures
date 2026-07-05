#pragma once

#include <stdexcept>
#include <string>
#include <cstddef>

enum class Status {
  Ok,
  UnsupportedDType,
  UnsupportedKernel,
  UnsupportedExecution,
  ValidationFailed,
};

enum class DType {
  Int32,
  Float32,
  Float16,
  BFloat16,
};

struct Tolerance {
  double abs;
  double rel;
};

inline DType parseDType(const std::string &value) {
  if (value == "int32") {
    return DType::Int32;
  }
  if (value == "float32") {
    return DType::Float32;
  }
  if (value == "float16") {
    return DType::Float16;
  }
  if (value == "bfloat16") {
    return DType::BFloat16;
  }
  throw std::runtime_error("Unsupported dtype: " + value);
}

inline const char *dtypeName(DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return "int32";
  case DType::Float32:
    return "float32";
  case DType::Float16:
    return "float16";
  case DType::BFloat16:
    return "bfloat16";
  }
  return "unknown";
}

inline std::size_t dtypeSize(DType dtype) {
  switch (dtype) {
  case DType::Int32:
  case DType::Float32:
    return 4;
  case DType::Float16:
  case DType::BFloat16:
    return 2;
  }
  throw std::runtime_error("Unsupported dtype");
}

inline Tolerance toleranceFor(DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return {0.0, 0.0};
  case DType::Float32:
    return {1e-5, 1e-5};
  case DType::Float16:
    return {1e-2, 1e-2};
  case DType::BFloat16:
    return {2e-2, 2e-2};
  }
  return {0.0, 0.0};
}

inline int defaultVectorWidth(DType dtype) {
  switch (dtype) {
  case DType::Int32:
  case DType::Float32:
    return 4;
  case DType::Float16:
  case DType::BFloat16:
    return 2;
  }
  return 1;
}
