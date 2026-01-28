#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    ASSERT(vals->ndim() == 1, "Argmax: vals must be a 1-D tensor.");
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1, "Argmax: max_idx and max_val must be single-element tensors.");

    // max_val should have same dtype as vals.
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // index tensor must be integer type (support at least I64 and I32)
    auto idx_dtype = max_idx->dtype();
    switch (idx_dtype) {
    case LLAISYS_DTYPE_I64:
    case LLAISYS_DTYPE_I32:
        break;
    default:
        ASSERT(false, "Argmax: unsupported index tensor dtype.");
    }

    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");

    // always support cpu calculation
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), idx_dtype, vals->numel());
    }

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());

    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), idx_dtype, vals->numel());
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