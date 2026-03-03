#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/sample_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/sample_nvidia.hpp"
#endif

namespace llaisys::ops {
void sample(tensor_t token, tensor_t logits, float temperature, int top_k, float top_p) {
    // 检查token张量：必须在CPU，类型I64，单个元素
    ASSERT(token->deviceType() == LLAISYS_DEVICE_CPU, "sample: output token tensor must be on CPU.");
    ASSERT(token->dtype() == LLAISYS_DTYPE_I64, "sample: output token tensor must have dtype I64.");
    ASSERT(token->numel() == 1, "sample: output token tensor must have exactly 1 element.");

    // 检查logits张量：必须是1D且连续
    ASSERT(logits->ndim() == 1, "sample: logits must be a 1-D tensor.");
    ASSERT(logits->isContiguous(), "sample: logits tensor must be contiguous.");

    size_t vocab_size = logits->shape()[0];

    // 如果logits在CPU上，直接调用CPU实现
    if (logits->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::sample_cpu(token->data(), logits->data(), logits->dtype(),
                        vocab_size, temperature, top_k, top_p);
        return;
    }

    // 设置设备上下文
    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());

    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::sample_cpu(token->data(), logits->data(), logits->dtype(),
                        vocab_size, temperature, top_k, top_p);
        break;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::sample_nvidia(token->data(), logits->data(), logits->dtype(),
                              vocab_size, temperature, top_k, top_p);
        break;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}