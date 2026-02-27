#include "add_nvidia.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {

// 通用模板，通过 reinterpret_cast 处理不同数据类型
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// FP16 特化：使用 __half 类型（确保 CUDA 架构 >= 6.0）
template<>
__global__ void add_kernel<__half>(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);   // 使用硬件 FP16 加法
    }
}

// BF16 特化：使用 __nv_bfloat16（需要 CUDA 11+ 且架构 >= Ampere）
template<>
__global__ void add_kernel<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);   // __nv_bfloat16 也支持 __hadd
    }
}

// 调度函数
void add(std::byte* c, const std::byte* a, const std::byte* b,
         llaisysDataType_t type, size_t numel) {
    // 选择网格/线程块大小
    const int block = 256;
    const int grid = (numel + block - 1) / block;

    switch (type) {
        case LLAISYS_DTYPE_F32:
            add_kernel<float><<<grid, block>>>(
                reinterpret_cast<const float*>(a),
                reinterpret_cast<const float*>(b),
                reinterpret_cast<float*>(c), numel);
            break;
        case LLAISYS_DTYPE_F16:
            add_kernel<__half><<<grid, block>>>(
                reinterpret_cast<const __half*>(a),
                reinterpret_cast<const __half*>(b),
                reinterpret_cast<__half*>(c), numel);
            break;
        case LLAISYS_DTYPE_BF16:
            add_kernel<__nv_bfloat16><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(a),
                reinterpret_cast<const __nv_bfloat16*>(b),
                reinterpret_cast<__nv_bfloat16*>(c), numel);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    // 同步（可选，上下文管理可能会自动同步）
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia