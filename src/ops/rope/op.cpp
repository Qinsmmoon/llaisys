#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#include "nvidia/rope_nvidia.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // in/out expected shape: [seqlen, nhead, d]
    ASSERT(in->ndim() == 3, "RoPE: in must be 3-D [seqlen, nhead, d].");
    ASSERT(out->ndim() == 3, "RoPE: out must be 3-D [seqlen, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1-D [seqlen].");

    const auto in_shape = in->shape();
    const auto out_shape = out->shape();
    const auto pid_shape = pos_ids->shape();

    ASSERT(in_shape.size() == 3 && out_shape.size() == 3 && pid_shape.size() == 1, "RoPE: invalid tensor shapes.");

    const size_t seqlen = in_shape[0];
    const size_t nhead = in_shape[1];
    const size_t d = in_shape[2];

    ASSERT(static_cast<size_t>(out_shape[0]) == seqlen && static_cast<size_t>(out_shape[1]) == nhead &&
           static_cast<size_t>(out_shape[2]) == d, "RoPE: out shape must equal in shape.");

    ASSERT(static_cast<size_t>(pid_shape[0]) == seqlen, "RoPE: pos_ids length must equal seqlen.");

    // d must be even because we pair dims
    ASSERT((d % 2) == 0, "RoPE: hidden dimension d must be even.");

    // pos_ids must be int64
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be Int64.");

    // in and out must have same dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    // dispatch to CPU implementation (always supported)
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
    }

    // set device and dispatch
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops