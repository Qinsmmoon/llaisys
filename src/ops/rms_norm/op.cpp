#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

// keep the header include (preferred) but also provide a local forward-declaration
// in case of include-order / build-system issues that made the CPU namespace invisible.
#include "cpu/rms_norm_cpu.hpp"
#include "nvidia/rms_norm_nvidia.hpp"

namespace llaisys::ops {
// forward-declare the cpu implementation to ensure the compiler sees the symbol
namespace cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype, size_t N, size_t d, float eps);
} // namespace cpu

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    // Shapes:
    // in: [N, d] (2-D)
    // out: [N, d] (2-D)
    // weight: [d] (1-D)
    ASSERT(in->ndim() == 2, "RMSNorm: input must be 2-D.");
    ASSERT(out->ndim() == 2, "RMSNorm: output must be 2-D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1-D.");

    const auto in_shape = in->shape();
    const auto out_shape = out->shape();
    const auto w_shape = weight->shape();

    ASSERT(in_shape.size() == 2 && out_shape.size() == 2 && w_shape.size() == 1, "RMSNorm: invalid tensor shapes.");

    const size_t N = static_cast<size_t>(in_shape[0]);
    const size_t d = static_cast<size_t>(in_shape[1]);

    ASSERT(static_cast<size_t>(out_shape[0]) == N && static_cast<size_t>(out_shape[1]) == d,
           "RMSNorm: out must have shape [in.shape(0), in.shape(1)].");
    ASSERT(static_cast<size_t>(w_shape[0]) == d, "RMSNorm: weight length must equal last dim of input.");

    // dtype checks: out, in, weight must have same dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

    // eps must be non-negative
    ASSERT(eps >= 0.0f, "RMSNorm: eps must be >= 0.");

    // dispatch to CPU implementation (always supported)
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), N, d, eps);
    }

    // else set device and dispatch
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), N, d, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), N, d, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops