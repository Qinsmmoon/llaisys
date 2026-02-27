#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
void rope_impl(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    const size_t half = d / 2;

    // Precompute denom = theta^(2*j/d) for j in [0, half)
    std::vector<float> denom(half);
    for (size_t j = 0; j < half; ++j) {
        // use double for pow for stability, then cast to float
        double exponent = 2.0 * static_cast<double>(j) / static_cast<double>(d);
        double val = std::pow(static_cast<double>(theta), exponent);
        denom[j] = static_cast<float>(val);
    }

    // iterate sequence and heads
    for (size_t i = 0; i < seqlen; ++i) {
        const int64_t p = pos_ids[i];
        for (size_t h = 0; h < nhead; ++h) {
            size_t base = (i * nhead + h) * d;
            for (size_t j = 0; j < half; ++j) {
                size_t ia = base + j;          // a component
                size_t ib = base + half + j;  // b component

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[ia]);
                    float b = llaisys::utils::cast<float>(in[ib]);

                    // phi = pos / denom[j]
                    float phi = static_cast<float>(p) / denom[j];
                    float c = std::cos(phi);  // 使用 std::cos 而不是 std::cosf
                    float s = std::sin(phi);  // 使用 std::sin 而不是 std::sinf

                    float oa = a * c - b * s;
                    float ob = b * c + a * s;

                    out[ia] = llaisys::utils::cast<T>(oa);
                    out[ib] = llaisys::utils::cast<T>(ob);
                } else {
                    // assume T is float (F32)
                    float a = static_cast<float>(in[ia]);
                    float b = static_cast<float>(in[ib]);

                    float phi = static_cast<float>(p) / denom[j];
                    float c = std::cos(phi);  // 使用 std::cos 而不是 std::cosf
                    float s = std::sin(phi);  // 使用 std::sin 而不是 std::sinf

                    float oa = a * c - b * s;
                    float ob = b * c + a * s;

                    out[ia] = static_cast<T>(oa);
                    out[ib] = static_cast<T>(ob);
                }
            }
        }
    }
}

void rope(std::byte *out_b, const std::byte *in_b, const std::byte *pos_b, llaisysDataType_t dtype,
          size_t seqlen, size_t nhead, size_t d, float theta) {
    // pos_ids is int64_t array
    const int64_t *pos_ids = reinterpret_cast<const int64_t *>(pos_b);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl(reinterpret_cast<float *>(out_b), reinterpret_cast<const float *>(in_b), pos_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl(reinterpret_cast<llaisys::fp16_t *>(out_b), reinterpret_cast<const llaisys::fp16_t *>(in_b), pos_ids, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl(reinterpret_cast<llaisys::bf16_t *>(out_b), reinterpret_cast<const llaisys::bf16_t *>(in_b), pos_ids, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu