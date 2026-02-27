#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
void linear_impl(T *out, const T *in, const T *weight, const T *bias, size_t N, size_t D_in, size_t D_out) {
    // weight is laid out as [D_out, D_in], we compute out[i,j] = dot(in[i, :], weight[j, :]) + bias[j]
    for (size_t i = 0; i < N; ++i) {
        const T *in_row = in + i * D_in;
        T *out_row = out + i * D_out;
        for (size_t j = 0; j < D_out; ++j) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                // accumulate in float for reduced precision types
                float acc = 0.0f;
                const T *w_row = weight + j * D_in;
                for (size_t k = 0; k < D_in; ++k) {
                    float a = llaisys::utils::cast<float>(in_row[k]);
                    float b = llaisys::utils::cast<float>(w_row[k]);
                    acc += a * b;
                }
                if (bias) acc += llaisys::utils::cast<float>(bias[j]);
                out_row[j] = llaisys::utils::cast<T>(acc);
            } else {
                // regular floating point (e.g., float)
                double acc = 0.0; // use double for accumulation safety
                const T *w_row = weight + j * D_in;
                for (size_t k = 0; k < D_in; ++k) {
                    acc += static_cast<double>(in_row[k]) * static_cast<double>(w_row[k]);
                }
                if (bias) acc += static_cast<double>(bias[j]);
                out_row[j] = static_cast<T>(acc);
            }
        }
    }
}

void linear(std::byte *out_b, const std::byte *in_b, const std::byte *weight_b, const std::byte *bias_b, llaisysDataType_t dtype,
            size_t N, size_t D_in, size_t D_out) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_impl(reinterpret_cast<float *>(out_b), reinterpret_cast<const float *>(in_b),
                           reinterpret_cast<const float *>(weight_b),
                           bias_b ? reinterpret_cast<const float *>(bias_b) : nullptr, N, D_in, D_out);
    case LLAISYS_DTYPE_F16:
        return linear_impl(reinterpret_cast<llaisys::fp16_t *>(out_b), reinterpret_cast<const llaisys::fp16_t *>(in_b),
                           reinterpret_cast<const llaisys::fp16_t *>(weight_b),
                           bias_b ? reinterpret_cast<const llaisys::fp16_t *>(bias_b) : nullptr, N, D_in, D_out);
    case LLAISYS_DTYPE_BF16:
        return linear_impl(reinterpret_cast<llaisys::bf16_t *>(out_b), reinterpret_cast<const llaisys::bf16_t *>(in_b),
                           reinterpret_cast<const llaisys::bf16_t *>(weight_b),
                           bias_b ? reinterpret_cast<const llaisys::bf16_t *>(bias_b) : nullptr, N, D_in, D_out);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu