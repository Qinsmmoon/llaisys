#include "embedding_nvidia.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {

// 通用 embedding kernel，T 可以是 float, __half, __nv_bfloat16
template<typename T>
__global__ void embedding_kernel(const int64_t* idx, const T* weight, T* out,
                                 size_t N, size_t weight_rows, size_t emb_dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * emb_dim) return;                     // 超出范围直接返回

    size_t out_row = i / emb_dim;                      // 当前输出元素属于哪个索引
    size_t col     = i % emb_dim;                      // 嵌入维度内的列偏移
    int64_t id = idx[out_row];                          // 读取索引值

    // 边界检查（与 CPU 版本行为一致，越界时赋 0；也可触发断言，但 kernel 内难以抛出异常）
    if (id >= 0 && static_cast<size_t>(id) < weight_rows) {
        out[i] = weight[id * emb_dim + col];
    } else {
        out[i] = T(0);                                  // 越界时填 0
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t N, size_t weight_rows, size_t emb_dim) {
    // 参数校验：必须存在有效数据
    ASSERT(N > 0 && emb_dim > 0 && weight_rows > 0, "Embedding: invalid dimensions");

    // 计算总元素数
    size_t numel = N * emb_dim;

    // 配置 kernel 启动参数（与 add_nvidia.cu 风格一致）
    const int block = 256;
    const int grid  = (numel + block - 1) / block;

    // 将输入指针转换为正确的类型
    const int64_t* idx_dev = reinterpret_cast<const int64_t*>(index);

    // 根据数据类型分派不同的 kernel
    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float* weight_dev = reinterpret_cast<const float*>(weight);
        float* out_dev = reinterpret_cast<float*>(out);
        embedding_kernel<float><<<grid, block>>>(idx_dev, weight_dev, out_dev, N, weight_rows, emb_dim);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        // 假设 llaisys::fp16_t 的内存布局与 __half 完全相同（都是 16 位）
        const __half* weight_dev = reinterpret_cast<const __half*>(weight);
        __half* out_dev = reinterpret_cast<__half*>(out);
        embedding_kernel<__half><<<grid, block>>>(idx_dev, weight_dev, out_dev, N, weight_rows, emb_dim);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        // 假设 llaisys::bf16_t 的内存布局与 __nv_bfloat16 完全相同（都是 16 位）
        const __nv_bfloat16* weight_dev = reinterpret_cast<const __nv_bfloat16*>(weight);
        __nv_bfloat16* out_dev = reinterpret_cast<__nv_bfloat16*>(out);
        embedding_kernel<__nv_bfloat16><<<grid, block>>>(idx_dev, weight_dev, out_dev, N, weight_rows, emb_dim);
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

    // 等待 kernel 完成（可选，根据上下文决定是否需要同步）
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia