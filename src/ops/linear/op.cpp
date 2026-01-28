#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    // bias may be nullptr or an empty tensor (no bias)

    // ranks
    ASSERT(in->ndim() == 2, "Linear: in must be 2-D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2-D.");
    ASSERT(out->ndim() == 2, "Linear: out must be 2-D.");

    const auto in_shape = in->shape();
    const auto weight_shape = weight->shape();
    const auto out_shape = out->shape();

    ASSERT(in_shape.size() == 2 && weight_shape.size() == 2 && out_shape.size() == 2,
           "Linear: unexpected tensor shapes.");

    // X: [N, D_in]
    const size_t N = static_cast<size_t>(in_shape[0]);
    const size_t D_in = static_cast<size_t>(in_shape[1]);

    // W: [D_out, D_in]  (note: weight is not transposed on disk)
    const size_t D_out = static_cast<size_t>(weight_shape[0]);
    const size_t W_cols = static_cast<size_t>(weight_shape[1]);

    ASSERT(W_cols == D_in, "Linear: weight.shape[1] must equal in.shape[1].");

    ASSERT(out_shape[0] == N && out_shape[1] == D_out, "Linear: out must be [in.shape(0), weight.shape(0)].");

    // dtype checks: out, in, weight must have same dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    bool has_bias = (bool)bias;
    if (has_bias) {
        // allow bias to be provided but empty (treated as no bias)
        if (bias->numel() == 0) {
            has_bias = false;
        } else {
            ASSERT(bias->ndim() == 1 && static_cast<size_t>(bias->numel()) == D_out,
                   "Linear: bias must be 1-D with length equal to weight.shape(0).");
            CHECK_SAME_DTYPE(bias->dtype(), out->dtype());
        }
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!has_bias || bias->isContiguous()),
           "Linear: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const std::byte *bias_ptr = has_bias ? bias->data() : nullptr;
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(), N, D_in, D_out);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU: {
        const std::byte *bias_ptr = has_bias ? bias->data() : nullptr;
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr, out->dtype(), N, D_in, D_out);
    }
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