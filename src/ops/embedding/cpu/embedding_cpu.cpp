#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace llaisys::ops::cpu {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype,
               size_t N, size_t weight_rows, size_t emb_dim) {
    // index is int64_t
    const int64_t *idx = reinterpret_cast<const int64_t *>(index);

    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float *weight_f = reinterpret_cast<const float *>(weight);
        float *out_f = reinterpret_cast<float *>(out);
        const size_t row_bytes = emb_dim * sizeof(float);
        for (size_t i = 0; i < N; ++i) {
            int64_t id = idx[i];
            ASSERT(id >= 0 && static_cast<size_t>(id) < weight_rows, "Embedding: index out of range.");
            const void *src = weight_f + (static_cast<size_t>(id) * emb_dim);
            void *dst = out_f + (i * emb_dim);
            std::memcpy(dst, src, row_bytes);
        }
        return;
    }
    case LLAISYS_DTYPE_F16: {
        const llaisys::fp16_t *weight_h = reinterpret_cast<const llaisys::fp16_t *>(weight);
        llaisys::fp16_t *out_h = reinterpret_cast<llaisys::fp16_t *>(out);
        const size_t row_bytes = emb_dim * sizeof(llaisys::fp16_t);
        for (size_t i = 0; i < N; ++i) {
            int64_t id = idx[i];
            ASSERT(id >= 0 && static_cast<size_t>(id) < weight_rows, "Embedding: index out of range.");
            const void *src = weight_h + (static_cast<size_t>(id) * emb_dim);
            void *dst = out_h + (i * emb_dim);
            std::memcpy(dst, src, row_bytes);
        }
        return;
    }
    case LLAISYS_DTYPE_BF16: {
        const llaisys::bf16_t *weight_b = reinterpret_cast<const llaisys::bf16_t *>(weight);
        llaisys::bf16_t *out_b = reinterpret_cast<llaisys::bf16_t *>(out);
        const size_t row_bytes = emb_dim * sizeof(llaisys::bf16_t);
        for (size_t i = 0; i < N; ++i) {
            int64_t id = idx[i];
            ASSERT(id >= 0 && static_cast<size_t>(id) < weight_rows, "Embedding: index out of range.");
            const void *src = weight_b + (static_cast<size_t>(id) * emb_dim);
            void *dst = out_b + (i * emb_dim);
            std::memcpy(dst, src, row_bytes);
        }
        return;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu