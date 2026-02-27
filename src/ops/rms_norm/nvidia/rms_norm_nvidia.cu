// rms_norm_nvidia.cpp
#include "rms_norm_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {

// 通用 kernel 模板（仅用于 float 类型）
template<typename T>
__global__ void rms_norm_kernel(const T* in, const T* weight, T* out,
                                 size_t N, size_t d, float eps) {
    extern __shared__ float shared_sumsq[];   // 规约用共享内存

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const T* row_in = in + bid * d;
    T* row_out = out + bid * d;

    // 第一阶段：计算当前线程的部分平方和
    float part_sumsq = 0.0f;
    for (size_t col = tid; col < d; col += blockDim.x) {
        float v = static_cast<float>(row_in[col]);
        part_sumsq += v * v;
    }
    shared_sumsq[tid] = part_sumsq;
    __syncthreads();

    // 块内规约
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sumsq[tid] += shared_sumsq[tid + offset];
        }
        __syncthreads();
    }

    float sumsq = shared_sumsq[0];
    __syncthreads();

    // 计算归一化系数
    float mean_sq = sumsq / static_cast<float>(d);
    float inv = rsqrtf(mean_sq + eps);   // 1 / sqrt(mean_sq + eps)

    // 第二阶段：应用归一化和缩放
    for (size_t col = tid; col < d; col += blockDim.x) {
        float v = static_cast<float>(row_in[col]);
        float w = static_cast<float>(weight[col]);
        float y = w * (v * inv);
        row_out[col] = static_cast<T>(y);
    }
}

// 特化：__half
template<>
__global__ void rms_norm_kernel<__half>(const __half* in, const __half* weight,
                                         __half* out, size_t N, size_t d, float eps) {
    extern __shared__ float shared_sumsq[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const __half* row_in = in + bid * d;
    __half* row_out = out + bid * d;

    float part_sumsq = 0.0f;
    for (size_t col = tid; col < d; col += blockDim.x) {
        float v = __half2float(row_in[col]);
        part_sumsq += v * v;
    }
    shared_sumsq[tid] = part_sumsq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sumsq[tid] += shared_sumsq[tid + offset];
        }
        __syncthreads();
    }

    float sumsq = shared_sumsq[0];
    __syncthreads();

    float mean_sq = sumsq / static_cast<float>(d);
    float inv = rsqrtf(mean_sq + eps);

    for (size_t col = tid; col < d; col += blockDim.x) {
        float v = __half2float(row_in[col]);
        float w = __half2float(weight[col]);
        float y = w * (v * inv);
        row_out[col] = __float2half(y);
    }
}

// 特化：__nv_bfloat16
template<>
__global__ void rms_norm_kernel<__nv_bfloat16>(const __nv_bfloat16* in,
                                                const __nv_bfloat16* weight,
                                                __nv_bfloat16* out,
                                                size_t N, size_t d, float eps) {
    extern __shared__ float shared_sumsq[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const __nv_bfloat16* row_in = in + bid * d;
    __nv_bfloat16* row_out = out + bid * d;

    float part_sumsq = 0.0f;
    for (size_t col = tid; col < d; col += blockDim.x) {
        float v = __bfloat162float(row_in[col]);
        part_sumsq += v * v;
    }
    shared_sumsq[tid] = part_sumsq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_sumsq[tid] += shared_sumsq[tid + offset];
        }
        __syncthreads();
    }

    float sumsq = shared_sumsq[0];
    __syncthreads();

    float mean_sq = sumsq / static_cast<float>(d);
    float inv = rsqrtf(mean_sq + eps);

    for (size_t col = tid; col < d; col += blockDim.x) {
        float v = __bfloat162float(row_in[col]);
        float w = __bfloat162float(weight[col]);
        float y = w * (v * inv);
        row_out[col] = __float2bfloat16(y);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t N, size_t d, float eps) {
    ASSERT(N > 0 && d > 0, "RMSNorm: invalid dimensions");

    const int block = 256;                     // 线程块大小
    const int grid = N;                         // 每行一个块
    size_t shared_mem = block * sizeof(float);  // 规约用共享内存

    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float* in_f32 = reinterpret_cast<const float*>(in);
        const float* w_f32  = reinterpret_cast<const float*>(weight);
        float* out_f32      = reinterpret_cast<float*>(out);
        rms_norm_kernel<float><<<grid, block, shared_mem>>>(in_f32, w_f32, out_f32, N, d, eps);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const __half* in_h = reinterpret_cast<const __half*>(in);
        const __half* w_h  = reinterpret_cast<const __half*>(weight);
        __half* out_h      = reinterpret_cast<__half*>(out);
        rms_norm_kernel<__half><<<grid, block, shared_mem>>>(in_h, w_h, out_h, N, d, eps);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const __nv_bfloat16* in_bf = reinterpret_cast<const __nv_bfloat16*>(in);
        const __nv_bfloat16* w_bf  = reinterpret_cast<const __nv_bfloat16*>(weight);
        __nv_bfloat16* out_bf      = reinterpret_cast<__nv_bfloat16*>(out);
        rms_norm_kernel<__nv_bfloat16><<<grid, block, shared_mem>>>(in_bf, w_bf, out_bf, N, d, eps);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia