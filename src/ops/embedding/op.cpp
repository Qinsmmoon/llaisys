#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2-D.");
    ASSERT(index->ndim() == 1, "Embedding: index must be 1-D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2-D.");

    // weight_shape = [vocab_size, embedding_dim]
    const auto weight_shape = weight->shape(); // vector<size_t>
    const auto index_shape = index->shape();
    const auto out_shape = out->shape();

    ASSERT(weight_shape.size() == 2, "Embedding: weight shape invalid.");
    ASSERT(index_shape.size() == 1, "Embedding: index shape invalid.");
    ASSERT(out_shape.size() == 2, "Embedding: out shape invalid.");

    // 索引数量
    const size_t N = static_cast<size_t>(index->numel());
    // 词表大小
    const size_t weight_rows = weight_shape[0];
    // 嵌入维度
    const size_t emb_dim = weight_shape[1];

    ASSERT(out_shape[0] == N && out_shape[1] == emb_dim,
           "Embedding: out shape must be [index_len, weight.shape(1)].");

    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be Int64.");

    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), N, weight_rows, emb_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), N, weight_rows, emb_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops