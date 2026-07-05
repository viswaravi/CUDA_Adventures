#pragma once

#include <cstddef>

namespace kernel_lab {

template <typename T> double hostReduceSum(const T *data, std::size_t length) {
    double sum = 0.0;
    for (std::size_t i = 0; i < length; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

} // namespace kernel_lab
