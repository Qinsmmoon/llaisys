#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    // Shapes expectations:
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3-D.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3-D.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3-D.");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3-D.");

    const auto q_shape = q->shape();
    const auto k_shape = k->shape();
    const auto v_shape = v->shape();
    const auto out_shape = attn_val->shape();

    ASSERT(q_shape.size() == 3 && k_shape.size() == 3 && v_shape.size() == 3 && out_shape.size() == 3,
           "SelfAttention: invalid tensor shapes.");

    const size_t seqlen = q_shape[0];
    const size_t nhead = q_shape[1];
    const size_t d = q_shape[2];

    const size_t total_len = k_shape[0];
    const size_t nkvhead = k_shape[1];
    const size_t kd = k_shape[2];

    const size_t v_total_len = v_shape[0];
    const size_t v_nkvhead = v_shape[1];
    const size_t dv = v_shape[2];

    ASSERT(kd == d, "SelfAttention: key hidden dim must equal q hidden dim.");
    ASSERT(total_len == v_total_len, "SelfAttention: k and v must have same leading length.");
    ASSERT(nkvhead == v_nkvhead, "SelfAttention: k and v must have same head count (nkvhead).");
    ASSERT(out_shape[0] == seqlen && out_shape[1] == nhead && out_shape[2] == dv,
           "SelfAttention: attn_val shape must be [seqlen, nhead, dv].");

    // dtype must match for q/k/v/attn_val
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), k->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), v->dtype());

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    // dispatch to CPU implementation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                   scale, seqlen, nhead, d, total_len, nkvhead, dv);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
                                   scale, seqlen, nhead, d, total_len, nkvhead, dv);
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