#include "linear_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {

// 通用线性层 kernel
template<typename T>
__global__ void linear_kernel(const T* in, const T* weight, const T* bias,
                              T* out, size_t N, size_t D_in, size_t D_out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D_out) return;

    size_t i = idx / D_out;          // batch index
    size_t j = idx % D_out;          // output feature index

    const T* in_row = in + i * D_in;
    const T* w_row  = weight + j * D_in;

    // 使用 float 累加，避免精度损失
    float acc = 0.0f;
    for (size_t k = 0; k < D_in; ++k) {
        float a = static_cast<float>(in_row[k]);
        float b = static_cast<float>(w_row[k]);
        acc += a * b;
    }
    if (bias) {
        acc += static_cast<float>(bias[j]);
    }

    out[i * D_out + j] = static_cast<T>(acc);
}

// 特化 __half：使用 CUDA 原生转换
template<>
__global__ void linear_kernel<__half>(const __half* in, const __half* weight,
                                       const __half* bias, __half* out,
                                       size_t N, size_t D_in, size_t D_out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D_out) return;

    size_t i = idx / D_out;
    size_t j = idx % D_out;

    const __half* in_row = in + i * D_in;
    const __half* w_row  = weight + j * D_in;

    float acc = 0.0f;
    for (size_t k = 0; k < D_in; ++k) {
        acc += __half2float(in_row[k]) * __half2float(w_row[k]);
    }
    if (bias) {
        acc += __half2float(bias[j]);
    }

    out[i * D_out + j] = __float2half(acc);
}

// 特化 __nv_bfloat16
template<>
__global__ void linear_kernel<__nv_bfloat16>(const __nv_bfloat16* in,
                                              const __nv_bfloat16* weight,
                                              const __nv_bfloat16* bias,
                                              __nv_bfloat16* out,
                                              size_t N, size_t D_in, size_t D_out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D_out) return;

    size_t i = idx / D_out;
    size_t j = idx % D_out;

    const __nv_bfloat16* in_row = in + i * D_in;
    const __nv_bfloat16* w_row  = weight + j * D_in;

    float acc = 0.0f;
    for (size_t k = 0; k < D_in; ++k) {
        acc += __bfloat162float(in_row[k]) * __bfloat162float(w_row[k]);
    }
    if (bias) {
        acc += __bfloat162float(bias[j]);
    }

    out[i * D_out + j] = __float2bfloat16(acc);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t N, size_t D_in, size_t D_out) {
    // 参数校验
    ASSERT(N > 0 && D_in > 0 && D_out > 0, "Linear: invalid dimensions");

    size_t numel = N * D_out;
    const int block = 256;
    const int grid = (numel + block - 1) / block;

    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float* in_f32   = reinterpret_cast<const float*>(in);
        const float* w_f32    = reinterpret_cast<const float*>(weight);
        const float* bias_f32 = bias ? reinterpret_cast<const float*>(bias) : nullptr;
        float* out_f32        = reinterpret_cast<float*>(out);
        linear_kernel<float><<<grid, block>>>(in_f32, w_f32, bias_f32, out_f32, N, D_in, D_out);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const __half* in_h   = reinterpret_cast<const __half*>(in);
        const __half* w_h    = reinterpret_cast<const __half*>(weight);
        const __half* bias_h = bias ? reinterpret_cast<const __half*>(bias) : nullptr;
        __half* out_h        = reinterpret_cast<__half*>(out);
        linear_kernel<__half><<<grid, block>>>(in_h, w_h, bias_h, out_h, N, D_in, D_out);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const __nv_bfloat16* in_bf   = reinterpret_cast<const __nv_bfloat16*>(in);
        const __nv_bfloat16* w_bf    = reinterpret_cast<const __nv_bfloat16*>(weight);
        const __nv_bfloat16* bias_bf = bias ? reinterpret_cast<const __nv_bfloat16*>(bias) : nullptr;
        __nv_bfloat16* out_bf        = reinterpret_cast<__nv_bfloat16*>(out);
        linear_kernel<__nv_bfloat16><<<grid, block>>>(in_bf, w_bf, bias_bf, out_bf, N, D_in, D_out);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    // 等待完成（可选）
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia