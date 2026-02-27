#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
void swiglu_impl(T *out, const T *gate, const T *up, size_t N, size_t D) {
    const size_t total = N * D;
    // elementwise: out = up * (gate * sigmoid(gate))  // SwiGLU 实际公式
    // SwiGLU = up * Swish(gate) 其中 Swish(x) = x * sigmoid(x)
    for (size_t i = 0; i < total; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float sigmoid = 1.0f / (1.0f + std::exp(-g));  // 使用 std::exp 而不是 std::expf
            float y = u * (g * sigmoid);
            out[i] = llaisys::utils::cast<T>(y);
        } else {
            float g = static_cast<float>(gate[i]);
            float u = static_cast<float>(up[i]);
            float sigmoid = 1.0f / (1.0f + std::exp(-g));  // 使用 std::exp 而不是 std::expf
            float y = u * (g * sigmoid);
            out[i] = static_cast<T>(y);
        }
    }
}

void swiglu(std::byte *out_b, const std::byte *gate_b, const std::byte *up_b, llaisysDataType_t dtype, size_t N, size_t D) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl(reinterpret_cast<float *>(out_b),
                           reinterpret_cast<const float *>(gate_b),
                           reinterpret_cast<const float *>(up_b),
                           N, D);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl(reinterpret_cast<llaisys::fp16_t *>(out_b),
                           reinterpret_cast<const llaisys::fp16_t *>(gate_b),
                           reinterpret_cast<const llaisys::fp16_t *>(up_b),
                           N, D);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl(reinterpret_cast<llaisys::bf16_t *>(out_b),
                           reinterpret_cast<const llaisys::bf16_t *>(gate_b),
                           reinterpret_cast<const llaisys::bf16_t *>(up_b),
                           N, D);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu