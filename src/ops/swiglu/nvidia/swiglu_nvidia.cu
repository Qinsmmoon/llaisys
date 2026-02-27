#include "swiglu_nvidia.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace llaisys::ops::nvidia {

// F32 kernel
__global__ void swiglu_kernel_f32(float *out, const float *gate, const float *up, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < total; i += stride) {
        float g = gate[i];
        float u = up[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i] = u * (g * sig);
    }
}

// FP16 kernel
__global__ void swiglu_kernel_f16(__half *out, const __half *gate, const __half *up, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < total; i += stride) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        float y = u * (g * sig);
        out[i] = __float2half(y);
    }
}

// BF16 kernel
__global__ void swiglu_kernel_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *gate, const __nv_bfloat16 *up, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < total; i += stride) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float sig = 1.0f / (1.0f + expf(-g));
        float y = u * (g * sig);
        out[i] = __float2bfloat16(y);
    }
}

void swiglu(std::byte *out_b, const std::byte *gate_b, const std::byte *up_b, llaisysDataType_t dtype, size_t N, size_t D) {
    ASSERT(N > 0 && D > 0, "SwiGLU: invalid dimensions");
    size_t total = N * D;

    const int block = 256;
    const int grid = static_cast<int>((total + block - 1) / block);

    cudaError_t err = cudaSuccess;

    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float *out_f = reinterpret_cast<float *>(out_b);
            const float *gate_f = reinterpret_cast<const float *>(gate_b);
            const float *up_f = reinterpret_cast<const float *>(up_b);
            swiglu_kernel_f32<<<grid, block>>>(out_f, gate_f, up_f, total);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            __half *out_h = reinterpret_cast<__half *>(out_b);
            const __half *gate_h = reinterpret_cast<const __half *>(gate_b);
            const __half *up_h = reinterpret_cast<const __half *>(up_b);
            swiglu_kernel_f16<<<grid, block>>>(out_h, gate_h, up_h, total);
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            __nv_bfloat16 *out_bf = reinterpret_cast<__nv_bfloat16 *>(out_b);
            const __nv_bfloat16 *gate_bf = reinterpret_cast<const __nv_bfloat16 *>(gate_b);
            const __nv_bfloat16 *up_bf = reinterpret_cast<const __nv_bfloat16 *>(up_b);
            swiglu_kernel_bf16<<<grid, block>>>(out_bf, gate_bf, up_bf, total);
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    // check launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    // synchronize and check execution errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia