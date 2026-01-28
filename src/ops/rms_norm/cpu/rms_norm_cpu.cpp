#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cstddef>
#include <cmath>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_impl(T *out, const T *in, const T *weight, size_t N, size_t d, float eps) {
    // For lower-precision types (bf16/fp16) we convert to float for accumulation
    for (size_t i = 0; i < N; ++i) {
        const T *in_row = in + i * d;
        T *out_row = out + i * d;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // accumulate squares in float
            float sumsq = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                float v = llaisys::utils::cast<float>(in_row[j]);
                sumsq += v * v;
            }
            float mean_sq = sumsq / static_cast<float>(d);
            float denom = std::sqrt(mean_sq + eps);
            float inv = 1.0f / denom;
            for (size_t j = 0; j < d; ++j) {
                float v = llaisys::utils::cast<float>(in_row[j]);
                float g = llaisys::utils::cast<float>(weight[j]);
                float y = g * (v * inv);
                out_row[j] = llaisys::utils::cast<T>(y);
            }
        } else {
            // use double accumulation for f32 for better numeric stability
            double sumsq = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double v = static_cast<double>(in_row[j]);
                sumsq += v * v;
            }
            double mean_sq = sumsq / static_cast<double>(d);
            double denom = std::sqrt(mean_sq + static_cast<double>(eps));
            double inv = 1.0 / denom;
            for (size_t j = 0; j < d; ++j) {
                double v = static_cast<double>(in_row[j]);
                double g = static_cast<double>(weight[j]);
                double y = g * (v * inv);
                out_row[j] = static_cast<T>(y);
            }
        }
    }
}

void rms_norm(std::byte *out_b, const std::byte *in_b, const std::byte *weight_b, llaisysDataType_t dtype, size_t N, size_t d, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl(reinterpret_cast<float *>(out_b), reinterpret_cast<const float *>(in_b),
                             reinterpret_cast<const float *>(weight_b), N, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl(reinterpret_cast<llaisys::fp16_t *>(out_b), reinterpret_cast<const llaisys::fp16_t *>(in_b),
                             reinterpret_cast<const llaisys::fp16_t *>(weight_b), N, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl(reinterpret_cast<llaisys::bf16_t *>(out_b), reinterpret_cast<const llaisys::bf16_t *>(in_b),
                             reinterpret_cast<const llaisys::bf16_t *>(weight_b), N, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu