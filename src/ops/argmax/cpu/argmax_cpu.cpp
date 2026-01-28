#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <limits>
#include <cmath>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
size_t argmax_index_and_store_value(std::byte *max_val_ptr, const std::byte *vals_ptr, size_t numel) {
    const T *vals = reinterpret_cast<const T *>(vals_ptr);
    // Assume numel > 0 (caller should ensure)
    size_t max_idx = 0;

    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // Cast to float for comparison
        float max_val_f = llaisys::utils::cast<float>(vals[0]);
        for (size_t i = 1; i < numel; ++i) {
            float v = llaisys::utils::cast<float>(vals[i]);
            if (v > max_val_f) {
                max_val_f = v;
                max_idx = i;
            }
        }
        // store max_val back as T
        T *out_val = reinterpret_cast<T *>(max_val_ptr);
        *out_val = llaisys::utils::cast<T>(max_val_f);
    } else {
        // Regular arithmetic type (e.g., float)
        T max_val = vals[0];
        for (size_t i = 1; i < numel; ++i) {
            if (vals[i] > max_val) {
                max_val = vals[i];
                max_idx = i;
            }
        }
        T *out_val = reinterpret_cast<T *>(max_val_ptr);
        *out_val = max_val;
    }

    return max_idx;
}

static void write_index_to_ptr(std::byte *idx_ptr, llaisysDataType_t idx_type, size_t idx_value) {
    switch (idx_type) {
    case LLAISYS_DTYPE_I64: {
        int64_t *p = reinterpret_cast<int64_t *>(idx_ptr);
        *p = static_cast<int64_t>(idx_value);
        return;
    }
    case LLAISYS_DTYPE_I32: {
        int32_t *p = reinterpret_cast<int32_t *>(idx_ptr);
        *p = static_cast<int32_t>(idx_value);
        return;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(idx_type);
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t val_type,
            llaisysDataType_t idx_type, size_t numel) {
    ASSERT(numel > 0, "Argmax: vals must contain at least one element.");

    size_t max_index = 0;
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        max_index = argmax_index_and_store_value<float>(max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        max_index = argmax_index_and_store_value<llaisys::bf16_t>(max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        max_index = argmax_index_and_store_value<llaisys::fp16_t>(max_val, vals, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }

    // write index into max_idx according to idx_type
    write_index_to_ptr(max_idx, idx_type, max_index);
}

} // namespace llaisys::ops::cpu