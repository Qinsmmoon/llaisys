#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_impl(T *out, const T *q, const T *k, const T *v,
                         float scale, size_t seqlen, size_t nhead, size_t d,
                         size_t total_len, size_t nkvhead, size_t dv) {
    // Map layout: element (t, h, c) stored at index ((t * head + h) * dim + c)
    const size_t q_row_stride = nhead * d;
    const size_t k_row_stride = nkvhead * d;
    const size_t v_row_stride = nkvhead * dv;
    const size_t out_row_stride = nhead * dv;

    const ptrdiff_t offset = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen);

    // If nkvhead != nhead emulate torch.repeat_interleave along head dim:
    size_t head_repeat = 1;
    if (nkvhead != nhead) {
        ASSERT((nhead % nkvhead) == 0, "self_attention: nhead must be a multiple of nkvhead for head repeat behavior.");
        head_repeat = nhead / nkvhead;
    }

    // Temporary per-head buffers (scores, exps) allocated once and reused.
    // We'll fill them entirely for each head to avoid stale values.
    std::vector<double> scores(total_len);
    std::vector<double> exps(total_len);

    for (size_t t = 0; t < seqlen; ++t) {
        const ptrdiff_t allowed_max = offset + static_cast<ptrdiff_t>(t);

        for (size_t h = 0; h < nhead; ++h) {
            // q pointer for this (t,h)
            const T *q_ptr = q + (t * q_row_stride) + (h * d);

            // compute dot products Q_t,h . K_{: , mapped_head}
            double max_score = -std::numeric_limits<double>::infinity();

            // fill scores for this head
            for (size_t ki = 0; ki < total_len; ++ki) {
                if (static_cast<ptrdiff_t>(ki) > allowed_max) {
                    scores[ki] = -std::numeric_limits<double>::infinity();
                    continue;
                }

                size_t k_head = (nkvhead == nhead) ? h : (h / head_repeat);
                const T *k_ptr = k + (ki * k_row_stride) + (k_head * d);

                // accumulate dot in double for stability
                double acc = 0.0;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t c = 0; c < d; ++c) {
                        double a = static_cast<double>(llaisys::utils::cast<float>(q_ptr[c]));
                        double b = static_cast<double>(llaisys::utils::cast<float>(k_ptr[c]));
                        acc += a * b;
                    }
                } else {
                    for (size_t c = 0; c < d; ++c) {
                        acc += static_cast<double>(q_ptr[c]) * static_cast<double>(k_ptr[c]);
                    }
                }
                acc *= static_cast<double>(scale);
                scores[ki] = acc;
                if (acc > max_score) max_score = acc;
            }

            // stable softmax in double
            double sum_exp = 0.0;
            for (size_t ki = 0; ki < total_len; ++ki) {
                if (scores[ki] == -std::numeric_limits<double>::infinity()) {
                    exps[ki] = 0.0;
                } else {
                    // Use exp(score - max_score) in double
                    double e = std::exp(scores[ki] - max_score);
                    exps[ki] = e;
                    sum_exp += e;
                }
            }

            // output pointer
            T *out_ptr = out + (t * out_row_stride) + (h * dv);

            if (sum_exp == 0.0) {
                // no allowed keys -> zero output
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t m = 0; m < dv; ++m) out_ptr[m] = llaisys::utils::cast<T>(0.0f);
                } else {
                    for (size_t m = 0; m < dv; ++m) out_ptr[m] = static_cast<T>(0.0f);
                }
                continue;
            }

            // accumulate weighted sum over V in double
            std::vector<double> accv(dv);
            std::fill(accv.begin(), accv.end(), 0.0);

            for (size_t ki = 0; ki < total_len; ++ki) {
                double w = exps[ki] / sum_exp;
                if (w == 0.0) continue;

                size_t v_head = (nkvhead == nhead) ? h : (h / head_repeat);
                const T *v_ptr = v + (ki * v_row_stride) + (v_head * dv);

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t m = 0; m < dv; ++m) {
                        accv[m] += w * static_cast<double>(llaisys::utils::cast<float>(v_ptr[m]));
                    }
                } else {
                    for (size_t m = 0; m < dv; ++m) {
                        accv[m] += w * static_cast<double>(v_ptr[m]);
                    }
                }
            }

            // write back (cast from double)
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for (size_t m = 0; m < dv; ++m) out_ptr[m] = llaisys::utils::cast<T>(static_cast<float>(accv[m]));
            } else {
                for (size_t m = 0; m < dv; ++m) out_ptr[m] = static_cast<T>(accv[m]);
            }
        } // end head
    } // end t
}

void self_attention(std::byte *attn_val_b, const std::byte *q_b, const std::byte *k_b, const std::byte *v_b,
                    llaisysDataType_t dtype, float scale,
                    size_t seqlen, size_t nhead, size_t d,
                    size_t total_len, size_t nkvhead, size_t dv) {
    if (nkvhead != nhead) {
        ASSERT((nhead % nkvhead) == 0, "self_attention (cpu): nhead must be a multiple of nkvhead when repeating heads.");
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl(reinterpret_cast<float *>(attn_val_b),
                                   reinterpret_cast<const float *>(q_b),
                                   reinterpret_cast<const float *>(k_b),
                                   reinterpret_cast<const float *>(v_b),
                                   scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl(reinterpret_cast<llaisys::fp16_t *>(attn_val_b),
                                   reinterpret_cast<const llaisys::fp16_t *>(q_b),
                                   reinterpret_cast<const llaisys::fp16_t *>(k_b),
                                   reinterpret_cast<const llaisys::fp16_t *>(v_b),
                                   scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl(reinterpret_cast<llaisys::bf16_t *>(attn_val_b),
                                   reinterpret_cast<const llaisys::bf16_t *>(q_b),
                                   reinterpret_cast<const llaisys::bf16_t *>(k_b),
                                   reinterpret_cast<const llaisys::bf16_t *>(v_b),
                                   scale, seqlen, nhead, d, total_len, nkvhead, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu