#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    // Expect 2-D tensors with identical shapes: [seqlen, intermediate_size]
    ASSERT(out->ndim() == 2, "SwiGLU: out must be 2-D.");
    ASSERT(gate->ndim() == 2, "SwiGLU: gate must be 2-D.");
    ASSERT(up->ndim() == 2, "SwiGLU: up must be 2-D.");

    const auto out_shape = out->shape();
    const auto gate_shape = gate->shape();
    const auto up_shape = up->shape();

    ASSERT(out_shape.size() == 2 && gate_shape.size() == 2 && up_shape.size() == 2,
           "SwiGLU: invalid tensor shapes.");
    ASSERT(out_shape == gate_shape && out_shape == up_shape, "SwiGLU: out/gate/up must have the same shape.");

    const size_t N = static_cast<size_t>(out_shape[0]);
    const size_t D = static_cast<size_t>(out_shape[1]);

    // dtype checks: out, gate, up must have same dtype
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype());
    CHECK_SAME_DTYPE(out->dtype(), up->dtype());

    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");

    // dispatch to CPU implementation (always supported)
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), N, D);
    }

    // else set device and dispatch
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), N, D);
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